import os
import random
import torch
import socket
import logging
import numpy as np
logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_cfg_gpu(cfg):
    """
     Setup arguments CUDA, GPU & distributed training
    """
    if cfg.LOCAL_RANK == -1 or cfg.NO_CUDA:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() and not cfg.NO_CUDA else "cpu")
        cfg.N_GPU = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(cfg.LOCAL_RANK)
        device = torch.device("cuda", cfg.LOCAL_RANK)
        if not torch.distributed.is_initialized:
            torch.distributed.init_process_group(backend="nccl")
        cfg.N_GPU = 1
    cfg.DEVICE = str(device)
    ws = os.environ.get('WORLD_SIZE')
    cfg.DISTRIBUTED_WORLD_SIZE = int(ws) if ws else 1

    logger.info(
        'Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d', socket.gethostname(),
        cfg.LOCAL_RANK, cfg.DEVICE,
        cfg.N_GPU,
        cfg.DISTRIBUTED_WORLD_SIZE)
    logger.info("16-bits training: %s ", cfg.FP16)
    return cfg
