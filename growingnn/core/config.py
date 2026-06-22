"""Project-wide defaults for growingnn."""

from __future__ import annotations
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

FLOAT_TYPE = np.float32
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper

RESHEPERS_CACHE_MAX_SIZE = 10
RESHEPERS_CACHE_MAX_MEMORY_MB = 100
RESHEPERS_CACHE_ENABLE_MONITORING = True

ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE = (0.0, 0.01)
RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE = "max"  # "avg" | "max"

# Properties for neuron deletion action
EDITABLE_MODULES = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d]
PASSTHROUGH_MODULES = (nn.Dropout, nn.Identity, nn.ReLU, nn.LeakyReLU,
                       nn.GELU, nn.SiLU, nn.Tanh, nn.ELU, nn.Sigmoid,
                       nn.MaxPool2d, nn.AvgPool2d,
                       nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                       nn.MaxPool1d, nn.AvgPool1d,
                       nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)
PASSTHROUGH_MODULES_TO_UPDATE = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
PASSTHROUGH_FUNCTIONS = frozenset({
    F.relu, F.gelu, F.silu, F.tanh, F.elu, F.sigmoid,
    torch.relu, torch.sigmoid, torch.tanh,
    torch.squeeze, torch.unsqueeze,
    "squeeze", "unsqueeze",
})
RESIZE_SAFE_MODULES = (nn.Linear,)

MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL = 5
MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE = 1_000_000
DEFAULT_NEURONS_SHRINK_RATIO = 0.5

TIME_EFFICIENCY_WEIGHT = 1.0
WEIGHT_COUNT_WEIGHT = 1e-6

# Monte Carlo tree search (architecture simulation)
MCTS_UCB1_C = 2
MCTS_ROLLOUT_DEPTH = 2
MCTS_ROLLOUT_EPOCHS = 1
MCTS_ROLLOUT_LR = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.0001)
#TODO: To be reaserched:
MCTS_UCB1_USE_SQRT = False  # False: legacy sum + log(N)/n; True: mean + sqrt(log(N)/n)
MCTS_PROPAGATE_ROLLOUT_VALUE = False  # False: return node.value; True: return latest rollout only

# LOGGING
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"  # str: NOTSET | DEBUG | INFO | WARNING | WARN | ERROR | CRITICAL; or int
LOG_TO_FILE = True
LOG_FILE_NAME = "growingnn.log"
LOG_FILE_MAX_BYTES = 100 * 1024 * 1024  # 100 MB per rotated file
LOG_FILE_BACKUP_COUNT = 9  # 1 active + 9 backups => ~1 GB on disk total

# DataLoader: subprocess count for loading batches (0 = main process only)
DATALOADER_NUM_WORKERS = 0


class RunningConfig:
    def __init__(self, 
        generations: int,
        epochs: int,
        lr_scheduler: LearningRateScheduler = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        stopper: TrainingStopper = TrainingStopper(StopperMode.EMPTY),
        #TODO: simualtion algs should also have parent type
        simulation_alg: Any | None = None,
        simulation_scheduler: SimulationScheduler = SimulationScheduler(SchedulerMode.NEVER),
        simulation_score: Any | None = None,
        simulation_set_size: int = 32,
        criterion: nn.Module | None = None,
        quiet: bool = False,
        print_every: int = 1):
        self.generations = generations
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.stopper = stopper
        self.simulation_alg = simulation_alg
        self.simulation_scheduler = simulation_scheduler
        self.simulation_score = simulation_score
        self.simulation_set_size = simulation_set_size
        self.criterion = criterion
        self.quiet = quiet
        self.print_every = print_every
        self.ACTIONS_ENABLE_ADD_SEQ_LAYER = True
        self.ACTIONS_ENABLE_ADD_RES_LAYER = True
        self.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER = True
        self.ACTIONS_ENABLE_ADD_RES_CONV_LAYER = True
        self.ACTIONS_ENABLE_DEL_LAYER = True
        self.ACTIONS_ENABLE_DEL_NEURONS_01 = True
        self.ACTIONS_ENABLE_DEL_NEURONS_05 = True
        self.ACTIONS_ENABLE_DEL_NEURONS_09 = True

    def set_simulation_loaders(self, sim_train_loader: DataLoader, sim_val_loader: DataLoader):
        self.sim_train_loader = sim_train_loader
        self.sim_val_loader = sim_val_loader

    def update_grow_actions(self, is_enabled: bool):
        self.ACTIONS_ENABLE_ADD_SEQ_LAYER = is_enabled
        self.ACTIONS_ENABLE_ADD_RES_LAYER = is_enabled
        self.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER = is_enabled
        self.ACTIONS_ENABLE_ADD_RES_CONV_LAYER = is_enabled

    def update_shrink_actions(self, is_enabled: bool):
        self.ACTIONS_ENABLE_DEL_LAYER = is_enabled
        self.ACTIONS_ENABLE_DEL_NEURONS_01 = is_enabled
        self.ACTIONS_ENABLE_DEL_NEURONS_05 = is_enabled
        self.ACTIONS_ENABLE_DEL_NEURONS_09 = is_enabled