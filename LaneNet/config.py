#!/usr/bin/env python3

import yacs.config

# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)

CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.seed = 0
_C.use_gpu = True
# Dataset
_C.DATASET = CN()
_C.DATASET.root = "data/tusimple"
_C.DATASET.train_txt = "train/train.txt"
_C.DATASET.val_txt = "train/val.txt"
_C.DATASET.test_txt = "test/test.txt"
_C.DATASET.val_ratio = 0.2
_C.DATASET.height = 256
_C.DATASET.width = 512
_C.DATASET.batch_size = 32
_C.DATASET.shuffle = True
_C.DATASET.num_workers = 4
# Train
_C.TRAIN = CN()
_C.TRAIN.max_epoch = 50
_C.TRAIN.print_freq = 5
_C.TRAIN.start_eval = 1
_C.TRAIN.eval_freq = 10
_C.TRAIN.optim = 'adam'
_C.TRAIN.lr = 3e-5
_C.TRAIN.weight_decay=5e-04
_C.TRAIN.momentum = 0.9
_C.TRAIN.adam_beta1 = 0.9
_C.TRAIN.adam_beta2 = 0.99
_C.TRAIN.lr_scheduler = 'single_step'
_C.TRAIN.step_size = 50
_C.TRAIN.gamma = 0.1
# 
_C.save_dir = 'log'


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config