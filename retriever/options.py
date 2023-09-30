import os
import gc
import random
import torch
import socket
import logging
import numpy as np
from torch.distributed import init_process_group
logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def report_cache_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    else:
        pass


def setup_cfg_gpu(cfg):
    """
     Setup arguments CUDA, GPU & distributed training
    """
    if cfg.DISTRIBUTED:
        init_process_group(backend="nccl")
        cfg.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cfg.LOCAL_RANK)
        device = torch.device("cuda", cfg.LOCAL_RANK)
        cfg.GLOBAL_RANK = int(os.environ["RANK"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.DEVICE = str(device)
    # cfg.DISTRIBUTED_WORLD_SIZE = torch.cuda.device_count()

    print(
        f"Initialized host {socket.gethostname()}, "
        f"local rank {cfg.LOCAL_RANK}, "
        f"global rank {cfg.GLOBAL_RANK}, "
        f"device={cfg.DEVICE}"
    )
    return cfg
