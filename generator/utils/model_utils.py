import collections
import logging
import glob
import os
import torch
import torch.nn as nn
from torch.serialization import default_restore_location
from torch.optim.lr_scheduler import LambdaLR
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import Adam, AdamW
from generator.models.gen_models import GENERATOR_T5

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
         'model_params',
         'optimizer_dict',
         'scheduler_dict',
         'step'
    ]
)

model_params_to_save = [
    'PROMPT_MAX_LENGTH',
    'ANSWER_MAX_LENGTH',
    'DO_LOWER_CASE'
]


def get_checkpoint_path(cfg, file_prefix) -> str:
    out_cp_files = glob.glob(os.path.join(cfg.GENERATOR.MODEL.MODEL_PATH, file_prefix + "*")) if cfg.GENERATOR.MODEL.MODEL_PATH else []
    logger.info("Checkpoint paths %s", out_cp_files)
    checkpoint_path = None

    if len(out_cp_files) > 0:
        checkpoint_path = max(out_cp_files, key=os.path.getctime)
        logger.info('Selected checkpoint path to load: %s', checkpoint_path)
    else:
        logger.info('No checkpoint file found at model path: %s', cfg.GENERATOR.MODEL.MODEL_PATH)
    return checkpoint_path


def load_states_from_checkpoint(checkpoint_path: str) -> CheckpointState:
    checkpoint_file_path = os.path.join(checkpoint_path, 'checkpoint.pth.tar')
    logger.info('Reading saved model from %s', checkpoint_file_path)
    state_dict = torch.load(checkpoint_file_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def set_model_cfg_from_state(state, cfg):
    if not state:
        return
    override_params = [(param, state[param]) for param in model_params_to_save if param in state]
    for param, value in override_params:
        if hasattr(cfg.GENERATOR.MODEL, param):
            logger.warning('Overriding cfg parameter value from checkpoint state. Param = %s, value = %s', param,
                           value)
        setattr(cfg.GENERATOR.MODEL, param, value)


def get_model_params_state(cfg):
    """
     Selects the param values to be saved in a checkpoint, so that a trained model faile can be used for downstream
     tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    r = {}
    for param in model_params_to_save:
        r[param] = getattr(cfg.GENERATOR.MODEL, param)
    return r


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def get_model_components(cfg, model_path=None):
    # modify based on cfg so it could load both bert and roberta models
    if 't5' in cfg.GENERATOR.MODEL.PRETRAINED_MODEL_TYPE:
        tokenizer = T5Tokenizer.from_pretrained(cfg.GENERATOR.MODEL.PRETRAINED_MODEL_TYPE)
        if not model_path:
            t5 = T5ForConditionalGeneration.from_pretrained(cfg.GENERATOR.MODEL.PRETRAINED_MODEL_TYPE)
            model = GENERATOR_T5(t5.config)
            model.load_t5(t5.state_dict())
        else:
            model = GENERATOR_T5.from_pretrained(model_path)
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
        optim_type=cfg.GENERATOR.SOLVER.OPTIMIZER.NAME,
        learning_rate=cfg.GENERATOR.SOLVER.OPTIMIZER.BASE_LR,
        adam_eps=cfg.GENERATOR.SOLVER.OPTIMIZER.EPS,
        weight_decay=cfg.GENERATOR.SOLVER.OPTIMIZER.WEIGHT_DECAY
    )
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup_steps=cfg.GENERATOR.SOLVER.OPTIMIZER.WARMUP_STEPS,
        total_training_steps=cfg.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS
    )
    return optimizer, scheduler


def setup_for_distributed_mode(
        model: nn.Module,
        optimizer: torch.optim.Optimizer, device: str, n_gpu: int = 1,
        LOCAL_RANK: int = -1,
        fp16: bool = False,
        fp16_opt_level: str = "O1"
) -> (nn.Module, torch.optim.Optimizer):
    model.to(device)
    if fp16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if optimizer is None:
            model = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if LOCAL_RANK != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],
                                                          output_device=LOCAL_RANK,
                                                          find_unused_parameters=True)
    return model, optimizer
