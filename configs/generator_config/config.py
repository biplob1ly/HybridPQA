import os
import warnings
from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode
from pathlib import Path

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# importing default as a global singleton
# cfg = _C
_C.DESCRIPTION = 'Default config from the Singleton'
_C.VERSION = 0
_C.OUTPUT_PATH = './output/'
_C.EXP = 'default'
_C.SEED = 42
_C.DEVICE = None
_C.LOCAL_RANK = -1
_C.DISTRIBUTED_WORLD_SIZE = None
_C.NO_CUDA = False
_C.N_GPU = None
_C.FP16 = False
# For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#        "See details at https://nvidia.github.io/apex/amp.html
_C.FP16_OPT_LEVEL = 'O1'


_C.GENERATOR = CfgNode()
_C.GENERATOR.DO_TRAIN = True
_C.GENERATOR.DO_TEST = True
# -----------------------------------------------------------------------------
# generator.DATA
# -----------------------------------------------------------------------------
_C.GENERATOR.DATA = CfgNode()
_C.GENERATOR.DATA.NAME = 'hetPQA'
_C.GENERATOR.DATA.DATA_PATH = './data/answer_generation/'
_C.GENERATOR.DATA.TRAIN_DATA_PATH = './data/answer_generation/train.json'
_C.GENERATOR.DATA.VAL_DATA_PATH = './data/answer_generation/dev.json'
_C.GENERATOR.DATA.TEST_DATA_PATH = './data/answer_generation/test.json'
_C.GENERATOR.DATA.NUM_CONTEXT = 3
_C.GENERATOR.DATA.INSERT_SOURCE = False
_C.GENERATOR.DATA.NORMALIZE = True
_C.GENERATOR.DATA.FLATTEN_ATTRIBUTE = True

# -----------------------------------------------------------------------------
# GENERATOR_MODEL
# -----------------------------------------------------------------------------
_C.GENERATOR.MODEL = CfgNode()
_C.GENERATOR.MODEL.MODEL_PATH = None
# config name for generator.MODEL initialization
_C.GENERATOR.MODEL.PRETRAINED_MODEL_TYPE = 't5-base'
# Max length of the encoder input sequence
_C.GENERATOR.MODEL.PROMPT_MAX_LENGTH = 200
_C.GENERATOR.MODEL.ANSWER_MAX_LENGTH = None
# Whether to lower case the input text. Set True for uncased generator.MODELs, False for the cased ones.
_C.GENERATOR.MODEL.DO_LOWER_CASE = True
_C.GENERATOR.MODEL.CHECKPOINT_FILE_NAME = 'generator_ckpt'
_C.GENERATOR.MODEL.DROPOUT = 0.1

_C.GENERATOR.SOLVER = CfgNode()
_C.GENERATOR.SOLVER.EVAL_ANSWER_MAX_LEN = 100
_C.GENERATOR.SOLVER.TRAIN_BATCH_SIZE = 2
_C.GENERATOR.SOLVER.VAL_BATCH_SIZE = 4
_C.GENERATOR.SOLVER.TEST_BATCH_SIZE = 1
_C.GENERATOR.SOLVER.TOTAL_TRAIN_STEPS = 1000
_C.GENERATOR.SOLVER.NUM_STEP_PER_EVAL = 300
_C.GENERATOR.SOLVER.CP_SAVE_LIMIT = 1
# Logging interval in terms of batches
_C.GENERATOR.SOLVER.RESET_CHECKPOINT_STEP = False

_C.GENERATOR.SOLVER.TEMPERATURE = 0.5

_C.GENERATOR.SOLVER.OPTIMIZER = CfgNode()
_C.GENERATOR.SOLVER.OPTIMIZER.NAME = 'AdamW'
# Linear warmup over warmup_steps.
_C.GENERATOR.SOLVER.OPTIMIZER.WARMUP_STEPS = 100
_C.GENERATOR.SOLVER.OPTIMIZER.BASE_LR = 1e-4
_C.GENERATOR.SOLVER.OPTIMIZER.EPS = 1e-8
_C.GENERATOR.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.01
_C.GENERATOR.SOLVER.OPTIMIZER.BETAS = (0.9, 0.999)
_C.GENERATOR.SOLVER.OPTIMIZER.RESET = False
_C.GENERATOR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS = 1
_C.GENERATOR.SOLVER.OPTIMIZER.MAX_GRAD_NORM = 1.0


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()


def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return list of them.
    # It is returning a list of hard overwrite.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == '':
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env.
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        "generator.DATA.TRAIN_DATA_PATH",
        "generator.DATA.VAL_DATA_PATH",
        "GENERATOR_MODEL.BACKBONE.PRETRAINED_PATH",
        "GENERATOR_SOLVER.LOSS.LABELS_WEIGHTS_PATH"
    }

    # Instantiate return list.
    path_overwrite_keys = []

    # Go through the list of key to be overwritten.
    for key in list_key_env:

        # Get value from the env.
        value = os.getenv("path_overwrite_keys")

        # If it is none, skip. As some keys are only needed during training and others during the prediction stage.
        if value is None:
            continue

        # Otherwise, adding the key and the value to the dictionary.
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys


def combine_cfgs(path_cfg_data: Path=None, path_cfg_override: Path=None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override=Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base