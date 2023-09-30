"""
train model
Usage:
    generator_trainer.py  --path_cfg_exp=<path> [--path_data=<path>] [--path_model=<path>] [--path_output=<path>] [--version=<val>] [--generator_ckpt=<filename>]
    generator_trainer.py -h | --help

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
    --generator_ckpt=<filename>       GENERATOR checkpoint file name
"""
from docopt import docopt
import os
import shutil
import time
from datetime import datetime
import numpy as np
import torch
from torch import Tensor as T
import logging
import random
from typing import Tuple
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from configs.generator_config.config import get_cfg_defaults

from generator.utils.model_utils import (
    get_checkpoint_path, get_model_components,
    get_optimizer_components, setup_for_distributed_mode,
    load_states_from_checkpoint, CheckpointState, get_model_obj,
    set_model_cfg_from_state, get_model_params_state
)
from generator.utils.data_utils import GenDataset, GenCollator
from generator.options import setup_cfg_gpu, set_seed
from generator_utils import BLEUScorer, GENERATORValResult, format_generator_validation, save_combined_results, save_eval_metrics, delete

logging.basicConfig(
    filename='generator_logs.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class GENERATORTrainer:
    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        self.shard_id = cfg.LOCAL_RANK if cfg.LOCAL_RANK != -1 else 0
        self.distributed_factor = cfg.DISTRIBUTED_WORLD_SIZE or 1
        saved_state = None
        if checkpoint_path:
            saved_state = load_states_from_checkpoint(checkpoint_path)
            set_model_cfg_from_state(saved_state.model_params, cfg)
        tokenizer, generator = get_model_components(cfg, checkpoint_path)
        optimizer, scheduler = get_optimizer_components(cfg, generator)
        generator, optimizer = setup_for_distributed_mode(generator, optimizer, cfg.DEVICE, cfg.N_GPU,
                                                      cfg.LOCAL_RANK,
                                                      cfg.FP16,
                                                      cfg.FP16_OPT_LEVEL)
        self.tokenizer = tokenizer
        self.generator = generator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_step = 0
        self.scheduler_state = None
        self.validations = []
        self.saved_cps = {}
        self.best_cp_name = None
        self.train_dataset = None
        self.val_dataset = None
        self.collator = GenCollator(tokenizer, cfg.GENERATOR.MODEL.PROMPT_MAX_LENGTH, cfg.GENERATOR.MODEL.ANSWER_MAX_LENGTH)
        self.eval_scorer = BLEUScorer()
        if saved_state:
            self._load_saved_state(saved_state)

    def evaluate(self, eval_dataset: GenDataset):
        logger.info('Evaluating generator ...')
        self.generator.eval()
        cfg = self.cfg
        eval_sampler = SequentialSampler(eval_dataset)
        eval_data_loader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=cfg.GENERATOR.SOLVER.TEST_BATCH_SIZE,
            drop_last=False,
            num_workers=1,
            collate_fn=self.collator
        )
        bleu_scores = []
        result_data = []
        with torch.no_grad():
            for iteration, batch in enumerate(eval_data_loader):
                model_outputs = self.generator.generate(
                    input_ids=batch.prompt_ids.to(cfg.DEVICE),
                    attention_mask=batch.prompt_masks.to(cfg.DEVICE),
                    max_length=cfg.GENERATOR.SOLVER.EVAL_ANSWER_MAX_LEN
                )
                for i, out_seq in enumerate(model_outputs):
                    pred_answer = self.tokenizer.decode(out_seq, skip_special_tokens=True)
                    data_example = eval_dataset.get_example(batch.indices[i])
                    gold_answers = data_example['answers']
                    score = self.eval_scorer.compute_bleu_score(gold_answers, pred_answer)
                    bleu_scores.append(score)
                    data_example['pred_answer'] = {'text': pred_answer, 'bleu': score}
                    result_data.append(data_example)
        mean_bleu_score = np.mean(bleu_scores)
        return mean_bleu_score, result_data

    def _save_checkpoint(self, step: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.generator)
        cp_dir = os.path.join(cfg.GENERATOR.MODEL.MODEL_PATH, cfg.GENERATOR.MODEL.CHECKPOINT_FILE_NAME + '.' + str(step))
        os.makedirs(cp_dir, exist_ok=True)
        model_to_save.save_pretrained(cp_dir)
        cp_fp = os.path.join(cp_dir, "checkpoint.pth.tar")

        meta_params = get_model_params_state(cfg)
        state = CheckpointState(
            meta_params,
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            step
        )
        torch.save(state._asdict(), cp_fp)
        logger.info('Saved checkpoint at %s', cp_fp)
        return cp_dir

    def validate_and_save(self, cur_step: int, val_dataset: GenDataset):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.LOCAL_RANK in [-1, 0]

        cur_val_id = len(self.validations)
        if cfg.GENERATOR.DATA.VAL_DATA_PATH:
            mean_bleu_score, _ = self.evaluate(val_dataset)
            val_metrics = ["bleu"]
            metrics_score = [mean_bleu_score]
            generator_eval = GENERATORValResult(cur_val_id, cur_step, val_metrics, metrics_score)
            self.validations.append(generator_eval)
            fmt_header, fmt_value = format_generator_validation(generator_eval)
            logger.info(fmt_header)
            logger.info(fmt_value)
            if cur_val_id == 0:
                print(fmt_header)
            print(fmt_value)

        if save_cp:
            best_generator_eval = max(self.validations, key=lambda x: x.scores)
            if len(self.saved_cps) < cfg.GENERATOR.SOLVER.CP_SAVE_LIMIT:
                cp_path = self._save_checkpoint(cur_step)
                self.saved_cps[cur_val_id] = cp_path
                if best_generator_eval.val_id == cur_val_id:
                    self.best_cp_name = cp_path
                    logger.info('New Best validation checkpoint %s', cp_path)
            else:
                sorted_generator_evals = sorted(self.validations, key=lambda x: x.scores, reverse=True)
                for generator_eval in sorted_generator_evals[cfg.GENERATOR.SOLVER.CP_SAVE_LIMIT:]:
                    if generator_eval.val_id in self.saved_cps:
                        delete(self.saved_cps[generator_eval.val_id])
                        del self.saved_cps[generator_eval.val_id]
                        cp_path = self._save_checkpoint(cur_step)
                        self.saved_cps[cur_val_id] = cp_path
                        if best_generator_eval.val_id == cur_val_id:
                            self.best_cp_name = cp_path
                            logger.info('New Best validation checkpoint %s', cp_path)
                        break

    def train(self, train_dataset, val_dataset=None):
        self.generator.train()
        cfg = self.cfg
        train_sampler = RandomSampler(train_dataset)
        train_data_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=cfg.GENERATOR.SOLVER.TRAIN_BATCH_SIZE,
            drop_last=True,
            num_workers=1,
            collate_fn=self.collator
        )

        logger.info("Total updates=%d", cfg.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS)
        logger.info(" Eval step = %d", cfg.GENERATOR.SOLVER.NUM_STEP_PER_EVAL)
        logger.info("***** Training *****")
        cur_step = self.start_step
        rolling_loss = 0
        epoch = 0
        last_saved_step = -1
        while cur_step < cfg.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS:
            epoch += 1
            logger.info("***** Epoch %d *****", epoch)
            for iteration, batch in enumerate(train_data_loader):
                model_outputs = self.generator(
                    input_ids=batch.prompt_ids.to(cfg.DEVICE),
                    attention_mask=batch.prompt_masks.to(cfg.DEVICE),
                    labels=batch.target_ids.to(cfg.DEVICE)
                )
                cur_loss = model_outputs.loss
                if self.cfg.GENERATOR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS > 1:
                    cur_loss = cur_loss / self.cfg.GENERATOR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS
                rolling_loss += cur_loss.item()
                cur_loss.backward()
                if (iteration + 1) % self.cfg.GENERATOR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), cfg.GENERATOR.SOLVER.OPTIMIZER.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.generator.zero_grad()
                    cur_step += 1

                if cur_step % cfg.GENERATOR.SOLVER.NUM_STEP_PER_EVAL == 0 and last_saved_step != cur_step:
                    logger.info(
                        "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
                        cfg.LOCAL_RANK,
                        cur_step,
                        cfg.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS,
                        rolling_loss/cfg.GENERATOR.SOLVER.NUM_STEP_PER_EVAL,
                        self.scheduler.get_last_lr()[0]
                    )
                    self.validate_and_save(cur_step, val_dataset)
                    self.generator.train()
                    rolling_loss = 0
                    last_saved_step = cur_step
                if cur_step >= cfg.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS:
                    break

        logger.info(
            "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
            cfg.LOCAL_RANK,
            cur_step,
            cfg.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS,
            rolling_loss / cfg.GENERATOR.SOLVER.NUM_STEP_PER_EVAL,
            self.scheduler.get_last_lr()[0]
        )
        self.validate_and_save(cur_step, val_dataset)
        logger.info("********** Training Completed **********")
        if cfg.LOCAL_RANK in [-1, 0]:
            for idx, generator_val_result in enumerate(self.validations):
                fmt_header, fmt_value = format_generator_validation(generator_val_result)
                if idx == 0:
                    logger.info(fmt_header)
                logger.info(fmt_value)
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)
        return self.best_cp_name

    def _load_saved_state(self, saved_state: CheckpointState):
        if self.cfg.GENERATOR.SOLVER.RESET_CHECKPOINT_STEP:
            self.step = 0
        else:
            self.step = saved_state.step

        if not self.cfg.GENERATOR.SOLVER.OPTIMIZER.RESET:
            if saved_state.optimizer_dict:
                logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(saved_state.optimizer_dict)
            if saved_state.scheduler_dict:
                logger.info("Loading scheduler state %s", saved_state.scheduler_dict)
                self.scheduler.load_state_dict(saved_state.scheduler_dict)


def run(cfg):
    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg.SEED)
    logger.info("***** Initializing model components *****")
    if cfg.GENERATOR.DO_TRAIN:
        cfg.GENERATOR.DATA.TRAIN_DATA_PATH = os.path.join(cfg.GENERATOR.DATA.DATA_PATH, 'mixed', 'train.json')
        cfg.GENERATOR.DATA.VAL_DATA_PATH = os.path.join(cfg.GENERATOR.DATA.DATA_PATH, 'mixed', 'dev.json')
        checkpoint_path = get_checkpoint_path(cfg, cfg.GENERATOR.MODEL.CHECKPOINT_FILE_NAME)
        generator_trainer = GENERATORTrainer(cfg, checkpoint_path=checkpoint_path)
        train_dataset = GenDataset(
            cfg.GENERATOR.DATA.TRAIN_DATA_PATH,
            n_context=cfg.GENERATOR.DATA.NUM_CONTEXT,
            normalize=cfg.GENERATOR.DATA.NORMALIZE,
            flatten_attr=cfg.GENERATOR.DATA.FLATTEN_ATTRIBUTE
        )
        val_dataset = GenDataset(
            cfg.GENERATOR.DATA.VAL_DATA_PATH,
            n_context=cfg.GENERATOR.DATA.NUM_CONTEXT,
            normalize=cfg.GENERATOR.DATA.NORMALIZE,
            flatten_attr=cfg.GENERATOR.DATA.FLATTEN_ATTRIBUTE
        )
        best_cp_path = generator_trainer.train(train_dataset, val_dataset=val_dataset)
        cfg.dump(stream=open(os.path.join(cfg.GENERATOR.MODEL.MODEL_PATH, f'config_{cfg.EXP}.yaml'), 'w'))
        cfg.GENERATOR.MODEL.CHECKPOINT_FILE_NAME = os.path.basename(best_cp_path)

    if cfg.GENERATOR.DO_TEST:
        cfg.GENERATOR.DATA.TEST_DATA_PATH = os.path.join(cfg.GENERATOR.DATA.DATA_PATH, 'mixed', 'test.json')
        checkpoint_path = get_checkpoint_path(cfg, cfg.GENERATOR.MODEL.CHECKPOINT_FILE_NAME)
        generator_trainer = GENERATORTrainer(cfg, checkpoint_path=checkpoint_path)
        test_dataset = GenDataset(
            cfg.GENERATOR.DATA.TEST_DATA_PATH,
            n_context=cfg.GENERATOR.DATA.NUM_CONTEXT,
            normalize=cfg.GENERATOR.DATA.NORMALIZE,
            flatten_attr=cfg.GENERATOR.DATA.FLATTEN_ATTRIBUTE
        )
        mean_bleu_score, result_data = generator_trainer.evaluate(test_dataset)
        combined_result_path = os.path.join(cfg.OUTPUT_PATH, 'combined_result.json')
        save_combined_results(result_data, combined_result_path)
        logger.info('Combined score saved in %s', combined_result_path)
        metrics_dt = {'BLEU': mean_bleu_score}
        eval_metrics_path = os.path.join(cfg.OUTPUT_PATH, f'eval_metrics')
        save_eval_metrics(metrics_dt, eval_metrics_path)
        logger.info('Evaluation done. Score per metric saved in %s', eval_metrics_path)


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    exp_cfg_path = arguments['--path_cfg_exp']
    data_path = arguments['--path_data']
    model_path = arguments['--path_model']
    output_path = arguments['--path_output']
    generator_ckpt = arguments['--generator_ckpt']
    version = arguments['--version']
    config = get_cfg_defaults()

    logger.info("Started logging...")
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if data_path is not None:
        config.GENERATOR.DATA.DATA_PATH = data_path
    if output_path is not None:
        config.OUTPUT_PATH = output_path
    if generator_ckpt is not None:
        config.GENERATOR.MODEL.CHECKPOINT_FILE_NAME = generator_ckpt
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
        config.GENERATOR.MODEL.MODEL_PATH = model_path
    else:
        config.GENERATOR.MODEL.MODEL_PATH = config.OUTPUT_PATH
    print(f'Model path: {config.GENERATOR.MODEL.MODEL_PATH}')
    logger.info(f'Model path: {config.GENERATOR.MODEL.MODEL_PATH}')
    run(config)
    shutil.copy(src='generator_logs.log', dst=os.path.join(config.OUTPUT_PATH, f'generator_logs_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}.log'))
