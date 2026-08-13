"""Efficiency-based simulation score terms."""

from __future__ import annotations

import time

import torch.fx as fx
import torch.nn as nn

import growingnn.core.config as config
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_training import (
    run_simulation_scoring_gradient_descent,
)
from growingnn.utils.fx import GraphStructureQuery


def score_time(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> float:
    start = time.time()
    run_simulation_scoring_gradient_descent(model, running_config)
    elapsed = time.time() - start
    return 1.0 / (config.TIME_EFFICIENCY_WEIGHT * elapsed + 1.0)


def score_count_weights(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> float:
    counter = GraphStructureQuery.get_amount_of_parameters(model)
    return 1.0 / (float(counter) * config.WEIGHT_COUNT_WEIGHT + 1.0)
