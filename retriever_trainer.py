"""
train model
Usage:
    retriever_trainer.py  --path_cfg_exp=<path> [--path_data=<path>] [--path_model=<path>] [--path_output=<path>] [--version=<val>] [--retriever_ckpt=<filename>]
    retriever_trainer.py -h | --help

Options:
    -h --help                   show this screen help
    --path_cfg_exp=<path>       experiment config path
    --path_data=<path>          data path
    --path_model=<path>         model path
    --path_output=<path>        output path
    --path_train_data=<path>    train data path
    --path_val_data=<path>      validation data path
    --path_test_data=<path>     Test data path
    --version=<val>             version
    --retriever_ckpt=<filename>       RETRIEVER checkpoint file name
"""
from docopt import docopt
import os
import shutil
from datetime import datetime
import torch
import logging
from timeit import default_timer as timer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from configs.retriever_config.config import get_cfg_defaults

from retriever.models.biencoder import Interaction, ContrastLoss
from retriever.utils.model_utils import (
    get_model_components, get_optimizer_components, get_model_obj,
    setup_for_distributed_mode, get_model_file, load_states_from_checkpoint,
    CheckpointState, set_model_cfg_from_state, get_model_params_state
)
from retriever.options import setup_cfg_gpu, set_seed
from retriever.utils.data_utils import RetDataset, CrossDataset, RetCollator, CrossTrainCollator, CrossEvalCollator, get_data_batch
from retriever_utils import (
    save_ranking_results, save_combined_results, save_eval_metrics, compute_metrics,
    RETRIEVERValResult, format_retriever_run, get_ranked_ctxs, get_weighted_ranked_ctxs
)
from calflops import calculate_flops

logging.basicConfig(
    filename='retriever_logs.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverTrainer:
    def __init__(self, cfg, model_file=None):
        self.cfg = cfg

        logger.info("***** Initializing model components *****")
        # if model file is specified, encoder parameters from saved state should be used for initialization
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_model_cfg_from_state(saved_state.model_params, cfg)

        tokenizer, biencoder = get_model_components(cfg)
        optimizer, scheduler = get_optimizer_components(cfg, biencoder)
        model, optimizer = setup_for_distributed_mode(cfg, biencoder, optimizer)
        self.idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
        self.tokenizer = tokenizer
        self.biencoder = biencoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_step = 0
        self.scheduler_state = None
        self.validations = []
        self.saved_cps = {}
        if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
            self.loss_function = None
        else:
            self.loss_function = ContrastLoss(
                encoder_type=cfg.RETRIEVER.MODEL.BIENCODER_TYPE,
                level=cfg.RETRIEVER.SOLVER.LEVEL,
                broadcast=cfg.RETRIEVER.SOLVER.BROADCAST,
                func=cfg.RETRIEVER.SOLVER.FUNC,
                temperature=cfg.RETRIEVER.SOLVER.TEMPERATURE,
                alpha=cfg.RETRIEVER.SOLVER.ALPHA,
                lambda_q=cfg.RETRIEVER.SOLVER.LAMBDA_QUERY,
                lambda_d=cfg.RETRIEVER.SOLVER.LAMBDA_CONTEXT,
                total_steps=cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS
            )
        if saved_state:
            # strict = not cfg.RETRIEVER.MODEL.PROJECTION_DIM
            self._load_saved_state(saved_state, strict=False)

    def get_representations(
            self,
            q_input_ids, q_attention_mask, q_token_type_ids,
            ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
    ):
        cfg = self.cfg
        device = cfg.DEVICE
        model_output = self.biencoder(
            q_input_ids=q_input_ids.to(device) if q_input_ids is not None else None,
            q_attention_mask=q_attention_mask.to(device) if q_attention_mask is not None else None,
            q_token_type_ids=q_token_type_ids.to(device) if q_token_type_ids is not None else None,
            ctx_input_ids=ctx_input_ids.to(device),
            ctx_attention_mask=ctx_attention_mask.to(device),
            ctx_token_type_ids=ctx_token_type_ids.to(device)
        )
        return model_output

    def visualize_sparse(self, batch):
        query_enc, ctx_enc = get_data_batch(batch, self.tokenizer)
        model_output = self.get_representations(
            query_enc['input_ids'], query_enc['attention_mask'], query_enc['token_type_ids'],
            ctx_enc['input_ids'], ctx_enc['attention_mask'], ctx_enc['token_type_ids']
        )
        q_sparse, d_sparse = model_output.q_sparse, model_output.d_sparse
        q_indices = q_sparse[0].nonzero().squeeze().tolist()
        q_values = q_sparse[0][q_indices].tolist()
        q_token2value = {self.idx2token[idx]: round(value, 2) for idx, value in zip(q_indices, q_values)}
        sorted_q_token2value = [f"{k: >7}: {v: >5.2f}" for k, v in sorted(q_token2value.items(), key=lambda item: item[1], reverse=True)[:50]]
        print(' | '.join(sorted_q_token2value))
        for i in range(d_sparse.size(0)):
            d_indices = d_sparse[i].nonzero().squeeze().tolist()
            d_values = d_sparse[i][d_indices].tolist()
            d_token2value = {self.idx2token[idx]: round(value, 2) for idx, value in zip(d_indices, d_values)}
            sorted_d_token2value = [f"{k: >7}: {v: >5.2f}" for k, v in sorted(d_token2value.items(), key=lambda item: item[1], reverse=True)[:80]]
            print(' | '.join(sorted_d_token2value))
        print('-'*30)

    def get_biencoder_result(self, batch, interaction):
        device = self.cfg.DEVICE
        sub_batch_size = self.cfg.RETRIEVER.SOLVER.TEST_CTX_BSZ
        alpha = self.cfg.RETRIEVER.SOLVER.ALPHA
        ctx_input_ids = batch.ctx_input_ids
        ctx_attention_mask = batch.ctx_attention_mask
        ctx_token_type_ids = batch.ctx_token_type_ids
        bsz = ctx_input_ids.size(0)
        exec_time = 0
        start = timer()
        q_sparse, q_dense = self.biencoder.encode_query(
            input_ids=batch.q_input_ids.to(device),
            attention_mask=batch.q_attention_mask.to(device),
            token_type_ids=batch.q_token_type_ids.to(device)
        )
        end = timer()
        exec_time += end - start
        if q_sparse is not None:
            q_sparse = q_sparse.detach().cpu()
        if q_dense is not None:
            q_dense = q_dense.detach().cpu()
        scores = {'sparse': [], 'dense': []}
        for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
            # Dim: sub_batch_size x S
            sub_ctx_input_ids = ctx_input_ids[batch_start:batch_start + sub_batch_size]
            sub_ctx_attention_mask = ctx_attention_mask[batch_start:batch_start + sub_batch_size]
            sub_ctx_token_type_ids = ctx_token_type_ids[batch_start:batch_start + sub_batch_size]

            ctx_sparse, ctx_dense = self.biencoder.encode_context(
                input_ids=sub_ctx_input_ids.to(device),
                attention_mask=sub_ctx_attention_mask.to(device),
                token_type_ids=sub_ctx_token_type_ids.to(device)
            )
            start = timer()
            if ctx_sparse is not None:
                ctx_sparse = ctx_sparse.detach().cpu()
                sparse_scores = interaction.compute_score(q_sparse, ctx_sparse)
                scores['sparse'].extend(sparse_scores.flatten().tolist())
            if ctx_dense is not None:
                ctx_dense = ctx_dense.detach().cpu()
                dense_scores = interaction.compute_score(q_dense, ctx_dense)
                scores['dense'].extend(dense_scores.flatten().tolist())
            end = timer()
            exec_time += end - start
        start = timer()
        scores, pred_ctx_ids, pred_ctx_srcs = get_weighted_ranked_ctxs(
            scores,
            batch.cids_per_qid[0],
            batch.srcs_per_qid[0],
            alpha=alpha
        )
        end = timer()
        exec_time += end - start
        result_dt = {
                'qid': batch.qids[0],
                'pred_ctx_sources': pred_ctx_srcs,
                'sparse_scores': scores['sparse'],
                'dense_scores': scores['dense'],
                'scores': scores['weighted'],
                'pred_ctx_ids': pred_ctx_ids,
                'actual_ctx_ids': batch.pos_cids_per_qid[0],
                'exec_time': exec_time
        }
        return result_dt

    def get_cross_encoder_result(self, batch):
        device = self.cfg.DEVICE
        sub_batch_size = self.cfg.RETRIEVER.SOLVER.TEST_CTX_BSZ
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        token_type_ids = batch.token_type_ids
        bsz = input_ids.size(0)
        scores = []
        start = timer()
        for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
            # Dim: sub_batch_size x S
            sub_input_ids = input_ids[batch_start:batch_start + sub_batch_size]
            sub_attention_mask = attention_mask[batch_start:batch_start + sub_batch_size]
            sub_token_type_ids = token_type_ids[batch_start:batch_start + sub_batch_size]
            model_output = self.biencoder(
                input_ids=sub_input_ids.to(device),
                attention_mask=sub_attention_mask.to(device),
                token_type_ids=sub_token_type_ids.to(device)
            )
            probabs = F.softmax(model_output.logits, dim=1)
            positive_probabs = probabs[:, 1].tolist()
            scores.extend(positive_probabs)

        ctx_scores_list, pred_ctx_ids, pred_ctx_srcs = get_ranked_ctxs(
            scores,
            batch.cids_per_qid[0],
            batch.srcs_per_qid[0]
        )
        end = timer()
        exec_time = end - start
        result_dt = {
            'qid': batch.qids[0],
            'pred_ctx_sources': pred_ctx_srcs,
            'scores': ctx_scores_list,
            'pred_ctx_ids': pred_ctx_ids,
            'actual_ctx_ids': batch.pos_cids_per_qid[0],
            'exec_time': exec_time
        }
        return result_dt

    def evaluate(self, eval_dataset: RetDataset):
        logger.info('Evaluating ranker ...')
        self.biencoder.eval()
        cfg = self.cfg
        eval_sampler = SequentialSampler(eval_dataset)
        if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
            collator = CrossEvalCollator(
                tokenizer=self.tokenizer,
                question_max_len=cfg.RETRIEVER.MODEL.QUESTION_MAX_LENGTH,
                ctx_max_len=cfg.RETRIEVER.MODEL.CONTEXT_MAX_LENGTH
            )
        else:
            collator = RetCollator(
                tokenizer=self.tokenizer,
                question_max_len=cfg.RETRIEVER.MODEL.QUESTION_MAX_LENGTH,
                ctx_max_len=cfg.RETRIEVER.MODEL.CONTEXT_MAX_LENGTH
            )
        eval_data_loader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=cfg.RETRIEVER.SOLVER.TEST_BATCH_SIZE,
            drop_last=False,
            collate_fn=collator
        )
        interaction = Interaction(
            level=cfg.RETRIEVER.SOLVER.LEVEL,
            broadcast=cfg.RETRIEVER.SOLVER.BROADCAST,
            func=cfg.RETRIEVER.SOLVER.FUNC
        )
        result_data = []
        exec_times = []
        ctx_times = []
        with torch.no_grad():
            for iteration, batch in enumerate(eval_data_loader):
                if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
                    result_dt = self.get_cross_encoder_result(batch)
                else:
                    result_dt = self.get_biencoder_result(batch, interaction)
                result_data.append(result_dt)
                exec_times.append(result_dt['exec_time'])
                ctx_times.append(result_dt['exec_time']/len(result_dt['pred_ctx_ids']))
        mean_exec_time = (sum(exec_times) / len(exec_times)) * 1000
        mean_ctx_time = (sum(ctx_times) / len(ctx_times)) * 1000
        print(f"Mean exec time: {mean_exec_time} ms, total query count: {len(exec_times)}, mean ctx time: {mean_ctx_time} ms")
        return result_data

    def _save_checkpoint(self, step: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(cfg.RETRIEVER.MODEL.MODEL_PATH,
                          cfg.RETRIEVER.MODEL.CHECKPOINT_FILE_NAME + '.' + str(step))

        meta_params = get_model_params_state(cfg)
        state = CheckpointState(
            model_to_save.state_dict(),
            meta_params,
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            step
        )
        torch.save(state._asdict(), cp)
        logger.info('Saved checkpoint at %s', cp)
        return cp

    def validate_and_save(self, cur_step: int, val_dataset: RetDataset):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.LOCAL_RANK in [-1, 0]

        cur_val_id = len(self.validations)
        if cfg.RETRIEVER.DATA.VAL_DATA_PATH:
            # validation_loss = self.validate_nll()
            result_list = self.evaluate(val_dataset)
            val_metrics = ["map", "r-precision", "mrr@5", "ndcg", "hit_rate@5", "precision@1"]
            metrics_dt = compute_metrics(result_list, val_metrics)['all']
            metrics_score = [metrics_dt[metric] for metric in val_metrics]
            ret_eval = RETRIEVERValResult(cur_val_id, cur_step, val_metrics, metrics_score)
            self.validations.append(ret_eval)
            fmt_header, fmt_value = format_retriever_run(ret_eval)
            logger.info(fmt_header)
            logger.info(fmt_value)
            if cur_val_id == 0:
                print(fmt_header)
            print(fmt_value)

        if save_cp:
            best_ret_eval = max(self.validations, key=lambda x: x.scores)
            if len(self.saved_cps) < cfg.RETRIEVER.SOLVER.CP_SAVE_LIMIT:
                cp_path = self._save_checkpoint(cur_step)
                self.saved_cps[cur_val_id] = cp_path
                if best_ret_eval.val_id == cur_val_id:
                    self.best_cp_name = cp_path
                    logger.info('New Best validation checkpoint %s', cp_path)
            else:
                sorted_runs = sorted(self.validations, key=lambda x: x.scores, reverse=True)
                for ret_eval in sorted_runs[cfg.RETRIEVER.SOLVER.CP_SAVE_LIMIT:]:
                    if ret_eval.val_id in self.saved_cps:
                        os.remove(self.saved_cps[ret_eval.val_id])
                        del self.saved_cps[ret_eval.val_id]
                        cp_path = self._save_checkpoint(cur_step)
                        self.saved_cps[cur_val_id] = cp_path
                        if best_ret_eval.val_id == cur_val_id:
                            self.best_cp_name = cp_path
                            logger.info('New Best validation checkpoint %s', cp_path)
                        break

    def calculate_loss(self, batch):
        cfg = self.cfg
        device = cfg.DEVICE
        if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
            model_output = self.biencoder(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                token_type_ids=batch.token_type_ids.to(device),
                labels=batch.labels.to(device)
            )
            cur_loss = model_output.loss
        else:
            model_output = self.biencoder(
                q_input_ids=batch.q_input_ids.to(device),
                q_attention_mask=batch.q_attention_mask.to(device),
                q_token_type_ids=batch.q_token_type_ids.to(device),
                ctx_input_ids=batch.ctx_input_ids.to(device),
                ctx_attention_mask=batch.ctx_attention_mask.to(device),
                ctx_token_type_ids=batch.ctx_token_type_ids.to(device)
            )
            cur_loss = self.loss_function.compute_loss(
                model_output=model_output,
                cids_per_qid=batch.cids_per_qid,
                pos_cids_per_qid=batch.pos_cids_per_qid
            )
        return cur_loss

    def train(self, train_dataset, val_dataset=None):
        self.biencoder.train()
        cfg = self.cfg
        train_sampler = RandomSampler(train_dataset)
        if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
            collator = CrossTrainCollator(
                tokenizer=self.tokenizer,
                question_max_len=cfg.RETRIEVER.MODEL.QUESTION_MAX_LENGTH,
                ctx_max_len=cfg.RETRIEVER.MODEL.CONTEXT_MAX_LENGTH
            )
        else:
            collator = RetCollator(
                tokenizer=self.tokenizer,
                question_max_len=cfg.RETRIEVER.MODEL.QUESTION_MAX_LENGTH,
                ctx_max_len=cfg.RETRIEVER.MODEL.CONTEXT_MAX_LENGTH
            )
        train_data_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=cfg.RETRIEVER.SOLVER.TRAIN_BATCH_SIZE,
            drop_last=True,
            collate_fn=collator
        )

        logger.info("Total updates=%d", cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS)
        logger.info("Eval step = %d", cfg.RETRIEVER.SOLVER.NUM_STEP_PER_EVAL)
        logger.info("***** Training *****")
        cur_step = self.start_step
        rolling_loss = 0
        epoch = 0
        last_saved_step = -1
        while cur_step < cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS:
            epoch += 1
            logger.info("***** Epoch %d *****", epoch)
            for iteration, batch in enumerate(train_data_loader):
                cur_loss = self.calculate_loss(batch)
                if self.cfg.RETRIEVER.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS > 1:
                    cur_loss = cur_loss / self.cfg.RETRIEVER.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS
                rolling_loss += cur_loss.item()
                cur_loss.backward()
                if (iteration + 1) % self.cfg.RETRIEVER.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), cfg.RETRIEVER.SOLVER.OPTIMIZER.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.biencoder.zero_grad()
                    cur_step += 1

                if cur_step % cfg.RETRIEVER.SOLVER.NUM_STEP_PER_EVAL == 0 and last_saved_step != cur_step:
                    logger.info(
                        "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
                        cfg.LOCAL_RANK,
                        cur_step,
                        cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS,
                        rolling_loss/cfg.RETRIEVER.SOLVER.NUM_STEP_PER_EVAL,
                        self.scheduler.get_last_lr()[0]
                    )
                    self.validate_and_save(cur_step, val_dataset)
                    self.biencoder.train()
                    rolling_loss = 0
                    last_saved_step = cur_step
                if cur_step >= cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS:
                    break

        logger.info(
            "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
            cfg.LOCAL_RANK,
            cur_step,
            cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS,
            rolling_loss / cfg.RETRIEVER.SOLVER.NUM_STEP_PER_EVAL,
            self.scheduler.get_last_lr()[0]
        )
        self.validate_and_save(cur_step, val_dataset)
        logger.info("********** Training Completed **********")
        if cfg.LOCAL_RANK in [-1, 0]:
            for idx, retriever_val_result in enumerate(self.validations):
                fmt_header, fmt_value = format_retriever_run(retriever_val_result)
                if idx == 0:
                    logger.info(fmt_header)
                logger.info(fmt_value)
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)
        return self.best_cp_name

    def _load_saved_state(self, saved_state: CheckpointState, strict=True):
        if self.cfg.RETRIEVER.SOLVER.RESET_CHECKPOINT_STEP:
            self.step = 0
            logger.info('Resetting checkpoint step=%s', self.step)
        else:
            self.step = saved_state.step
            logger.info('Loading checkpoint step=%s', self.step)

        model_to_load = get_model_obj(self.biencoder)
        logger.info('Loading saved model state ...')
        missing_keys, unexpected_keys = model_to_load.load_state_dict(saved_state.model_dict, strict=strict)
        # strict = False => it means that we just load the parameters of layers which are present in both and
        # ignores the rest
        if len(missing_keys) > 0:
            print("MISSING KEYS WHILE RESTORING THE MODEL")
            print(missing_keys)
        if len(unexpected_keys) > 0:
            print("UNEXPECTED KEYS WHILE RESTORING THE MODEL")
            print(unexpected_keys)
        print("restoring model:", model_to_load.__class__.__name__)

        if not self.cfg.RETRIEVER.SOLVER.OPTIMIZER.RESET:
            if saved_state.optimizer_dict:
                logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(saved_state.optimizer_dict)
            if saved_state.scheduler_dict:
                logger.info("Loading scheduler state %s", saved_state.scheduler_dict)
                self.scheduler.load_state_dict(saved_state.scheduler_dict)


def run(cfg):
    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg.SEED)

    if cfg.RETRIEVER.DO_TRAIN:
        train_file = 'processed/train.csv' if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross' else 'separated/norm_train.json'
        cfg.RETRIEVER.DATA.TRAIN_DATA_PATH = os.path.join(cfg.RETRIEVER.DATA.DATA_PATH, train_file)
        cfg.RETRIEVER.DATA.VAL_DATA_PATH = os.path.join(cfg.RETRIEVER.DATA.DATA_PATH, 'mixed', 'norm_dev.json')
        model_file = get_model_file(cfg)
        retriever_trainer = RetrieverTrainer(cfg, model_file=model_file)
        if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
            train_dataset = CrossDataset(
                file=cfg.RETRIEVER.DATA.TRAIN_DATA_PATH,
                normalize=cfg.RETRIEVER.DATA.NORMALIZE,
                flatten_attr=cfg.RETRIEVER.DATA.FLATTEN_ATTRIBUTE,
                split="train",
                count=cfg.RETRIEVER.DATA.COUNT
            )
        else:
            train_dataset = RetDataset(
                file=cfg.RETRIEVER.DATA.TRAIN_DATA_PATH,
                num_pos_ctx=cfg.RETRIEVER.DATA.NUM_POSITIVE_CONTEXTS,
                num_total_ctx=cfg.RETRIEVER.DATA.NUM_TOTAL_CONTEXTS,
                normalize=cfg.RETRIEVER.DATA.NORMALIZE,
                flatten_attr=cfg.RETRIEVER.DATA.FLATTEN_ATTRIBUTE,
                split="train",
                count=cfg.RETRIEVER.DATA.COUNT
            )
        val_dataset = RetDataset(
            file=cfg.RETRIEVER.DATA.VAL_DATA_PATH,
            num_pos_ctx=None,
            num_total_ctx=None,
            normalize=cfg.RETRIEVER.DATA.NORMALIZE,
            flatten_attr=cfg.RETRIEVER.DATA.FLATTEN_ATTRIBUTE,
            split="validation",
            count=cfg.RETRIEVER.DATA.COUNT
        )
        best_cp_path = retriever_trainer.train(train_dataset, val_dataset=val_dataset)
        cfg.dump(stream=open(os.path.join(cfg.RETRIEVER.MODEL.MODEL_PATH, f'config_{cfg.EXP}.yaml'), 'w'))
        cfg.RETRIEVER.MODEL.CHECKPOINT_FILE_NAME = os.path.basename(best_cp_path)

    if cfg.RETRIEVER.DO_TEST:
        cfg.RETRIEVER.DATA.TEST_DATA_PATH = os.path.join(cfg.RETRIEVER.DATA.DATA_PATH, 'mixed', 'fixed_norm_test.json')
        model_file = get_model_file(cfg)
        retriever_trainer = RetrieverTrainer(cfg, model_file=model_file)
        test_dataset = RetDataset(
            file=cfg.RETRIEVER.DATA.TEST_DATA_PATH,
            num_pos_ctx=None,
            num_total_ctx=None,
            normalize=cfg.RETRIEVER.DATA.NORMALIZE,
            flatten_attr=cfg.RETRIEVER.DATA.FLATTEN_ATTRIBUTE,
            split="test",
            count=cfg.RETRIEVER.DATA.COUNT
        )
        result_list = retriever_trainer.evaluate(test_dataset)
        ranking_result_path = os.path.join(cfg.OUTPUT_PATH, 'rank_score_ids.jsonl')
        save_ranking_results(result_list, ranking_result_path)
        logger.info('Rank and score saved in %s', ranking_result_path)
        combined_result_path = os.path.join(cfg.OUTPUT_PATH, 'combined_score_ids.json')
        save_combined_results(result_list, cfg.RETRIEVER.DATA.TEST_DATA_PATH, combined_result_path)
        logger.info('Combined score saved in %s', combined_result_path)
        eval_metrics = ["map", "r-precision", "mrr@5", "ndcg", "hit_rate@5", "precision@1"]
        metrics_dt = compute_metrics(result_list, eval_metrics, comp_separate=True)
        eval_metrics_path = os.path.join(cfg.OUTPUT_PATH, 'eval_metrics')
        save_eval_metrics(metrics_dt, eval_metrics_path)
        logger.info('Evaluation done. Score per metric saved in %s', eval_metrics_path)

    if cfg.RETRIEVER.COMPUTE_FLOPS:
        compute_FLOPs(cfg)

    if cfg.RETRIEVER.VISUALIZE:
        visualize(cfg)


def visualize(cfg):
    data = [
        {
            "qid": "2108",
            "question": "how long does a typical game take?",
            "positive_ctxs": [
                {
                    "cid": "19511",
                    "source": "review",
                    "text": "5-6 hours is probably a fairly accurate estimate of how long an actual game will take."
                }
            ],
            "negative_ctxs": []
        }
    ]
    model_file = get_model_file(cfg)
    retriever_trainer = RetrieverTrainer(cfg, model_file=model_file)
    retriever_trainer.visualize_sparse(data)


def compute_FLOPs(cfg):
    tokenizer, model = get_model_components(cfg)
    question = ["how many aaa batteries does this require?"]
    context = ["It requires 3 aaa batteries."]

    if cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
        max_seq_length = cfg.RETRIEVER.MODEL.QUESTION_MAX_LENGTH + cfg.RETRIEVER.MODEL.CONTEXT_MAX_LENGTH
        inputs = tokenizer(
            question,
            context,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        flops, macs, params = calculate_flops(
            model=model,
            kwargs=inputs,
            print_results=False
        )
    else:
        max_seq_length = cfg.RETRIEVER.MODEL.QUESTION_MAX_LENGTH
        inputs = tokenizer(
            question,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        flops, macs, params = calculate_flops(
            model=model.question_model,
            kwargs=inputs,
            print_results=False
        )
    # bert-base-uncased - cross encoder FLOPs : 45.94 GFLOPS, MACs: 22.95 GMACs, Params: 109.48 M
    # bert-base-uncased - hybrid FLOPs: 28.51 GFLOPS, MACs: 14.25 GMACs, Params: 109.51 M
    # bert-base-uncased - sparse lexical FLOPs: 28.51 GFLOPS, MACs: 14.25 GMACs, Params: 109.51 M
    # bert-base-uncased - independent dense FLOPs : 22.36 GFLOPS, MACs: 11.17 GMACs, Params: 109.48 M
    # bert-base-uncased - dense colbert FLOPs : 22.39 GFLOPS, MACs: 11.19 GMACs, Params: 109.58 M
    # bert-base-uncased - dense colber w/o projection FLOPs: 22.36 GFLOPS, MACs: 11.17 GMACs, Params: 109.48 M
    print(f"{cfg.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE} - {cfg.RETRIEVER.MODEL.BIENCODER_TYPE} FLOPs: {flops}, MACs: {macs}, Params: {params}")


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    exp_cfg_path = arguments['--path_cfg_exp']
    data_path = arguments['--path_data']
    model_path = arguments['--path_model']
    output_path = arguments['--path_output']
    retriever_ckpt = arguments['--retriever_ckpt']
    version = arguments['--version']
    config = get_cfg_defaults()

    logger.info("Started logging...")
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if data_path is not None:
        config.RETRIEVER.DATA.DATA_PATH = data_path
    if output_path is not None:
        config.OUTPUT_PATH = output_path
    if retriever_ckpt is not None:
        config.RETRIEVER.MODEL.CHECKPOINT_FILE_NAME = retriever_ckpt
    if version is None:
        version = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        logger.info(f"Version: {version}")

    # Make result folders if they do not exist
    config.OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, config.EXP, version)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH, exist_ok=False)
    print(f'Output path: {config.OUTPUT_PATH}')
    logger.info(f'Output path: {config.OUTPUT_PATH}')
    if model_path is not None:
        config.RETRIEVER.MODEL.MODEL_PATH = model_path
    else:
        config.RETRIEVER.MODEL.MODEL_PATH = config.OUTPUT_PATH
    print(f'Model path: {config.RETRIEVER.MODEL.MODEL_PATH}')
    logger.info(f'Model path: {config.RETRIEVER.MODEL.MODEL_PATH}')
    run(config)
    shutil.copy(src='retriever_logs.log', dst=os.path.join(config.OUTPUT_PATH, f'retriever_logs_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}.log'))
