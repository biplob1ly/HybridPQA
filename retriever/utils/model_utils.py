import collections
import logging
import glob
import os
import string
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.serialization import default_restore_location
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam, AdamW
from retriever.models.biencoder import BiEncoder, HybridEncoder
from retriever.models.hf_models import HFBertEncoder, HFBertEncoderForMLM
from transformers import BertTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger()
CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        'model_dict',
        'model_params',
        'optimizer_dict',
        'scheduler_dict',
        'step'
    ]
)

model_params_to_save = [
    'BIENCODER_TYPE',
    'PRETRAINED_MODEL_TYPE',
    'CROSS_INTERACTION',
    'POOLING',
    'PROJECTION_DIM',
    'QUESTION_MAX_LENGTH',
    'CONTEXT_MAX_LENGTH',
    'DO_LOWER_CASE'
]


def get_model_params_state(cfg):
    """
     Selects the param values to be saved in a checkpoint, so that a trained model faile can be used for downstream
     tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    r = {}
    for param in model_params_to_save:
        r[param] = getattr(cfg.RETRIEVER.MODEL, param)
    return r


def set_model_cfg_from_state(state, cfg):
    if not state:
        return
    override_params = [(param, state[param]) for param in model_params_to_save if param in state]
    for param, value in override_params:
        if hasattr(cfg.RETRIEVER.MODEL, param):
            logger.warning('Overriding cfg parameter value from checkpoint state. Param = %s, value = %s', param,
                           value)
        setattr(cfg.RETRIEVER.MODEL, param, value)


def get_skip_list(tokenizer):
    puncts = set(string.punctuation) - {'"', "'", '-', ':'}
    skip_items = list(puncts) + stopwords.words('english')
    skip_list = {w: True
            for symbol in skip_items
            for w in [symbol, tokenizer.encode(symbol, add_special_tokens=False)[0]]}
    special_tokens = [
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.sep_token_id
    ]
    return skip_list, special_tokens


def get_skip_mask(tokenizer):
    skip_words = list(string.punctuation) + tokenizer.all_special_tokens + stopwords.words('english')
    skip_tokens = set()
    for word in skip_words:
        tokens = tokenizer.tokenize(word)
        skip_tokens.update(tokens)
    skip_tokens = list(skip_tokens - {':', '-', '"'})
    token2idx = tokenizer.get_vocab()
    mask_indices = torch.tensor([token2idx[token] for token in skip_tokens])
    skip_mask = torch.ones(len(token2idx)).index_fill(dim=-1, index=mask_indices, value=0)
    return skip_mask


def get_model_components(cfg):
    if cfg.RETRIEVER.MODEL.BIENCODER_TYPE in ['hybrid', 'sparse', 'dense'] and 'bert' in cfg.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE:
        tokenizer = BertTokenizer.from_pretrained(cfg.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE,
                                                  do_lower_case=cfg.RETRIEVER.MODEL.DO_LOWER_CASE)

        if cfg.RETRIEVER.MODEL.BIENCODER_TYPE in ['hybrid', 'sparse']:
            skip_mask = get_skip_mask(tokenizer).to(cfg.DEVICE)
            question_encoder = HFBertEncoderForMLM(cfg.RETRIEVER.MODEL, skip_mask)
            ctx_encoder = HFBertEncoderForMLM(cfg.RETRIEVER.MODEL, skip_mask)
            model = BiEncoder(question_encoder, ctx_encoder, cfg.RETRIEVER.MODEL.BIENCODER_TYPE)
        else:
            skip_list, special_tokens = get_skip_list(tokenizer)
            question_encoder = HFBertEncoder(cfg.RETRIEVER.MODEL, [], special_tokens)
            ctx_encoder = HFBertEncoder(cfg.RETRIEVER.MODEL, skip_list, special_tokens)
            model = BiEncoder(question_encoder, ctx_encoder, cfg.RETRIEVER.MODEL.BIENCODER_TYPE)
    elif cfg.RETRIEVER.MODEL.BIENCODER_TYPE == 'cross':
        tokenizer = AutoTokenizer.from_pretrained(cfg.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE,
                                                  do_lower_case=cfg.RETRIEVER.MODEL.DO_LOWER_CASE)
        config = AutoConfig.from_pretrained(cfg.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE,
                                            num_labels=2,
                                            output_attentions=False,
                                            output_hidden_states=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE,
            config=config
        )
    else:
        raise NotImplementedError
    return tokenizer, model


def get_optimizer(
        model: nn.Module,
        optim_type: str,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optim_type == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps, weight_decay=weight_decay)
    elif optim_type == 'Adam':
        optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(
    optimizer,
    warmup_steps,
    total_training_steps,
    last_epoch=-1,
    fixed_lr=False
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if fixed_lr:
            return 1.0
        return max(
            1e-5,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optimizer_components(cfg, model):
    optimizer = get_optimizer(
        model=model,
        optim_type=cfg.RETRIEVER.SOLVER.OPTIMIZER.NAME,
        learning_rate=cfg.RETRIEVER.SOLVER.OPTIMIZER.BASE_LR,
        adam_eps=cfg.RETRIEVER.SOLVER.OPTIMIZER.EPS,
        weight_decay=cfg.RETRIEVER.SOLVER.OPTIMIZER.WEIGHT_DECAY
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_steps=cfg.RETRIEVER.SOLVER.OPTIMIZER.WARMUP_STEPS,
        total_training_steps=cfg.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS
    )
    return optimizer, scheduler


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def get_model_file(cfg) -> str:
    out_cp_files = glob.glob(os.path.join(cfg.RETRIEVER.MODEL.MODEL_PATH, cfg.RETRIEVER.MODEL.CHECKPOINT_FILE_NAME + "*")) if cfg.RETRIEVER.MODEL.MODEL_PATH else []
    logger.info("Checkpoint files %s", out_cp_files)
    model_file = None

    if len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
        logger.info('Selected file to load: %s', model_file)
    else:
        logger.info('No checkpoint file found at model path: %s', cfg.RETRIEVER.MODEL.MODEL_PATH)
    return model_file


def load_states_from_checkpoint(checkpoint_file_path: str) -> CheckpointState:
    logger.info('Reading saved model from %s', checkpoint_file_path)
    state_dict = torch.load(checkpoint_file_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def setup_for_distributed_mode(
        cfg,
        model,
        optimizer
):
    model.to(cfg.DEVICE)
    if cfg.FP16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if optimizer is None:
            model = amp.initialize(model, optimizer, opt_level=cfg.FP16_OPT_LEVEL)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.FP16_OPT_LEVEL)

    if cfg.DISTRIBUTED:
        model = DDP(model, device_ids=[cfg.LOCAL_RANK], output_device=cfg.LOCAL_RANK)  # find_unused_parameters=True
    return model, optimizer


def get_schedule_linear(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):

    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)