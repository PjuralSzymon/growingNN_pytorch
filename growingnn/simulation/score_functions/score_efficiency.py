"""Efficiency-based simulation score terms."""

from __future__ import annotations

import time

import torch.fx as fx
import torch.nn as nn

import growingnn.core.config as config
from growingnn.simulation.context import SimulationContext
from growingnn.training.gradient_descent import gradient_descent
from growingnn.utils.fx import GraphStructureQuery


def score_time(
    model: nn.Module | fx.GraphModule,
    ctx: SimulationContext,
) -> float:
    start = time.time()
    gradient_descent(
        model,
        ctx.epochs,
        ctx.train_loader,
        ctx.val_loader,
        ctx.criterion,
        ctx.lr_scheduler,
        quiet=True,
    )
    elapsed = time.time() - start
    return 1.0 / (config.TIME_EFFICIENCY_WEIGHT * elapsed + 1.0)


def score_count_weights(
    model: nn.Module | fx.GraphModule,
    ctx: SimulationContext,
) -> float:
    counter = GraphStructureQuery.get_amount_of_parameters(model)
    return 1.0 / (float(counter) * config.WEIGHT_COUNT_WEIGHT + 1.0)
