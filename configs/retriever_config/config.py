"""
config test
Usage:
    config.py --path_output=<path> --path_data=<path> [--path_train_data=<path>] [--path_val_data=<path>] [--path_cfg_exp=<path>]
    config.py -h | --help

Options:
    -h --help                   show this screen help
    --path_output=<path>        output path
    --path_data=<path>          data path
    --path_train_data=<path>    train data path
    --path_val_data=<path>      validation data path
    --path_cfg_exp=<path>       experiment config path
"""
# [default: configs/data.yaml]
import os
import warnings
from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode
from pathlib import Path
from docopt import docopt

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# importing default as a global singleton
# cfg = _C
_C.PROJECT = 'RETRIEVER'
_C.DESCRIPTION = 'Default config from the Singleton'
_C.VERSION = 0
_C.OUTPUT_PATH = './output/'
_C.EXP = 'default'
_C.SEED = 42
_C.DEVICE = None
_C.DISTRIBUTED = False
_C.LOCAL_RANK = -1
_C.GLOBAL_RANK = -1
_C.DISTRIBUTED_WORLD_SIZE = None
_C.LOG_ALL_PROCESS = False
_C.FP16 = False
# For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#        "See details at https://nvidia.github.io/apex/amp.html
_C.FP16_OPT_LEVEL = 'O1'


_C.RETRIEVER = CfgNode()
_C.RETRIEVER.DO_TRAIN = True
_C.RETRIEVER.DO_TEST = True
_C.RETRIEVER.COMPUTE_FLOPS = False
_C.RETRIEVER.VISUALIZE = False
# -----------------------------------------------------------------------------
# RETRIEVER.DATA
# -----------------------------------------------------------------------------
_C.RETRIEVER.DATA = CfgNode()
_C.RETRIEVER.DATA.NAME = 'hetPQA'
_C.RETRIEVER.DATA.DATA_PATH = './data/evidence_ranking/'
_C.RETRIEVER.DATA.TRAIN_DATA_PATH = './data/evidence_ranking/train.json'
_C.RETRIEVER.DATA.VAL_DATA_PATH = './data/evidence_ranking/dev.json'
_C.RETRIEVER.DATA.TEST_DATA_PATH = './data/evidence_ranking/test.json'
_C.RETRIEVER.DATA.NUM_POSITIVE_CONTEXTS = 1
_C.RETRIEVER.DATA.NUM_TOTAL_CONTEXTS = 5
_C.RETRIEVER.DATA.INSERT_SOURCE = True
_C.RETRIEVER.DATA.NORMALIZE = True
_C.RETRIEVER.DATA.FLATTEN_ATTRIBUTE = True
_C.RETRIEVER.DATA.COUNT = None
# -----------------------------------------------------------------------------
# RETRIEVER_MODEL
# -----------------------------------------------------------------------------
_C.RETRIEVER.MODEL = CfgNode()
_C.RETRIEVER.MODEL.MODEL_PATH = None
# config name for RETRIEVER.MODEL initialization
_C.RETRIEVER.MODEL.PRETRAINED_MODEL_TYPE = 'bert-base-uncased'
_C.RETRIEVER.MODEL.BIENCODER_TYPE = 'hybrid'
_C.RETRIEVER.MODEL.CROSS_INTERACTION = False
# Extra linear layer on top of standard bert/roberta encoder
_C.RETRIEVER.MODEL.POOLING = 'cls'
_C.RETRIEVER.MODEL.PROJECTION_DIM = None
# Max length of the encoder input sequence
_C.RETRIEVER.MODEL.QUESTION_MAX_LENGTH = 128
_C.RETRIEVER.MODEL.CONTEXT_MAX_LENGTH = 128
_C.RETRIEVER.MODEL.AGG = 'max'
_C.RETRIEVER.MODEL.TOP_K = 768
# Whether to lower case the input text. Set True for uncased RETRIEVER.MODELs, False for the cased ones.
_C.RETRIEVER.MODEL.DO_LOWER_CASE = True
_C.RETRIEVER.MODEL.CHECKPOINT_FILE_NAME = 'retriever_ckpt'
_C.RETRIEVER.MODEL.DROPOUT = 0.1

_C.RETRIEVER.SOLVER = CfgNode()
_C.RETRIEVER.SOLVER.TRAIN_BATCH_SIZE = 2
_C.RETRIEVER.SOLVER.TEST_BATCH_SIZE = 1
_C.RETRIEVER.SOLVER.TEST_CTX_BSZ = 32
_C.RETRIEVER.SOLVER.TOTAL_TRAIN_STEPS = 1000
_C.RETRIEVER.SOLVER.NUM_STEP_PER_EVAL = 300
_C.RETRIEVER.SOLVER.CP_SAVE_LIMIT = 1
# Logging interval in terms of batches
_C.RETRIEVER.SOLVER.RESET_CHECKPOINT_STEP = False

_C.RETRIEVER.SOLVER.LEVEL = 'pooled'
_C.RETRIEVER.SOLVER.BROADCAST = 'local'
_C.RETRIEVER.SOLVER.FUNC = 'cosine'
_C.RETRIEVER.SOLVER.TEMPERATURE = 0.5
_C.RETRIEVER.SOLVER.ALPHA = 0.5
_C.RETRIEVER.SOLVER.LAMBDA_QUERY = 0.0003
_C.RETRIEVER.SOLVER.LAMBDA_CONTEXT = 0.0001


_C.RETRIEVER.SOLVER.OPTIMIZER = CfgNode()
_C.RETRIEVER.SOLVER.OPTIMIZER.NAME = 'AdamW'
# Linear warmup over warmup_steps.
_C.RETRIEVER.SOLVER.OPTIMIZER.WARMUP_STEPS = 200
_C.RETRIEVER.SOLVER.OPTIMIZER.BASE_LR = 1e-5
_C.RETRIEVER.SOLVER.OPTIMIZER.EPS = 1e-8
_C.RETRIEVER.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.00
_C.RETRIEVER.SOLVER.OPTIMIZER.BETAS = (0.9, 0.999)
_C.RETRIEVER.SOLVER.OPTIMIZER.RESET = False
_C.RETRIEVER.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS = 8
_C.RETRIEVER.SOLVER.OPTIMIZER.MAX_GRAD_NORM = 1.0

# Amount of top docs to return
_C.RETRIEVER.SOLVER.TOP_RETRIEVE_COUNT = 5
# Temporal memory data buffer size (in samples) for indexer
_C.RETRIEVER.SOLVER.INDEX_BUFFER_SIZE = 500


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
        "RETRIEVER.DATA.TRAIN_DATA_PATH",
        "RETRIEVER.DATA.VAL_DATA_PATH",
        "RETRIEVER_MODEL.BACKBONE.PRETRAINED_PATH",
        "RETRIEVER_SOLVER.LOSS.LABELS_WEIGHTS_PATH"
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


if __name__ == '__main__':
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['--path_output']
    data_path = arguments['--path_data']
    train_data_path = arguments['--path_train_data']
    val_data_path = arguments['--path_val_data']
    exp_cfg_path = arguments['--path_cfg_exp']
    config = get_cfg_defaults()
    if data_path is not None:
        config.RETRIEVER.DATA.DATA_PATH = data_path
        config.RETRIEVER.DATA.TRAIN_DATA_PATH = os.path.join(data_path, 'train.json')
        config.RETRIEVER.DATA.VAL_DATA_PATH = os.path.join(data_path, 'dev.json')
    if train_data_path is not None:
        config.RETRIEVER.DATA.TRAIN_DATA_PATH = train_data_path
    if val_data_path is not None:
        config.RETRIEVER.DATA.VAL_DATA_PATH = val_data_path
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if output_path is not None:
        config.OUTPUT_PATH = output_path

    # Make result folders if they do not exist
    print(config)
    # config.dump(stream=open(os.path.join(exp_dir, f'config_{config.EXP}.yaml'), 'w'))
    # python - m src.tools.train - o experiments/exp10 --cfg src/config/experiments/exp10.yaml