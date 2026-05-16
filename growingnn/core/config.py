"""Project-wide defaults for growingnn."""

import numpy as np
from torch import nn

FLOAT_TYPE = np.float32

RESHEPERS_CACHE_MAX_SIZE = 10
RESHEPERS_CACHE_MAX_MEMORY_MB = 100
RESHEPERS_CACHE_ENABLE_MONITORING = True

ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE = (0.0, 0.01)
RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE = "max"  # "avg" | "max"

EDITABLE_MODULES = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d]

# LOGGING
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"  # str: NOTSET | DEBUG | INFO | WARNING | WARN | ERROR | CRITICAL; or int
LOG_TO_FILE = True
LOG_FILE_NAME = "growingnn.log"
LOG_FILE_MAX_BYTES = 100 * 1024 * 1024  # 100 MB per rotated file
LOG_FILE_BACKUP_COUNT = 9  # 1 active + 9 backups => ~1 GB on disk total