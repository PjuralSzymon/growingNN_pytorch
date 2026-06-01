"""Learning-based simulation score terms."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from growingnn.simulation.context import SimulationContext
from growingnn.training.gradient_descent import gradient_descent


def score_acc(
    model: nn.Module | fx.GraphModule,
    ctx: SimulationContext,
) -> float:
    cfg = ctx.running_config
    _, history = gradient_descent(
        model,
        cfg.simulation_scheduler.simulation_epochs,
        #TODO: is this a simualtion dataset ? or a training dataset ?
        ctx.train_loader,
        ctx.val_loader,
        ctx.criterion,
        cfg.lr_scheduler,
        quiet=True,
    )
    return float(history["val_acc"][-1]) # TODO: Orginally it was by train acc 


def score_loss(
    model: nn.Module | fx.GraphModule,
    ctx: SimulationContext,
) -> float:
    cfg = ctx.running_config
    _, history = gradient_descent(
        model,
        cfg.simulation_scheduler.simulation_epochs,
        #TODO: is this a simualtion dataset ? or a training dataset ?
        ctx.train_loader,
        ctx.val_loader,
        ctx.criterion,
        cfg.lr_scheduler,
        quiet=True,
    )
    loss = float(history["val_loss"][-1])
    return min(1.0 / max(loss, 1e-8), 1.0) #TODO: Orginally it was by train loss 
