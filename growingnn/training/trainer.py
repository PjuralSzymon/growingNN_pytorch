"""Generation loop: train, optionally simulate an architecture mutation, repeat."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from growingnn.actions.registry import generate_all_actions
from growingnn.core.logger import logger
from growingnn.simulation.context import SimulationContext
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler import LearningRateScheduler
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def train_generations(
    model: nn.Module | fx.GraphModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    lr_scheduler: LearningRateScheduler,
    *,
    generations: int,
    epochs: int,
    stopper: TrainingStopper | None = None,
    simulation_alg: Any | None = None,
    simulation_scheduler: SimulationScheduler | None = None,
    simulation_score: Any | None = None,
    simulation_set_size: int = 32,
    quiet: bool = True,
    print_every: int = 10,
) -> tuple[nn.Module | fx.GraphModule, dict[str, list[Any]]]:
    logger.info(f"Training generations started")
    if generations <= 0:
        raise ValueError("generations must be positive")

    stopper = stopper or TrainingStopper(StopperMode.EMPTY)
    simulation_scheduler = simulation_scheduler or SimulationScheduler(SchedulerMode.NEVER)
    sim_train_loader, sim_val_loader = sample_loaders(
        train_loader, val_loader, simulation_set_size
    )
    sim_ctx = SimulationContext(
        train_loader=sim_train_loader,
        val_loader=sim_val_loader,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        epochs=simulation_scheduler.simulation_epochs,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    generation_val_acc: list[float] = []
    combined: dict[str, list[Any]] = {
        "generation": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "param_count": [],
    }

    for generation in range(generations):
        logger.info(f"Training generation {generation} started")
        model, history = gradient_descent(
            model,
            epochs,
            train_loader,
            val_loader,
            criterion,
            lr_scheduler,
            stopper=stopper,
            quiet=quiet,
            print_every=print_every,
        )
        val_acc = history["val_acc"][-1]
        generation_val_acc.append(val_acc)
        combined["generation"].append(generation)
        combined["train_loss"].extend(history["train_loss"])
        combined["train_acc"].extend(history["train_acc"])
        combined["val_loss"].extend(history["val_loss"])
        combined["val_acc"].extend(history["val_acc"])
        combined["lr"].extend(history["lr"])
        param_count = GraphStructureQuery.get_amount_of_parameters(model)
        combined["param_count"].extend([param_count] * len(history["train_loss"]))

        metrics = {"accuracy": history["train_acc"][-1], "val_acc": val_acc}
        if stopper.check(model, generation, metrics):
            break

        is_last = generation >= generations - 1
        if is_last or simulation_alg is None or simulation_score is None:
            continue
        if not simulation_scheduler.can_simulate(generation, generation_val_acc, quiet=quiet):
            continue

        action, _, _ = loop.run_until_complete(
            simulation_alg.get_action(
                copy.deepcopy(model),
                simulation_scheduler.simulation_time,
                sim_ctx,
                simulation_score,
            )
        )
        clear_reshepers_cache()
        if action is None:
            continue
        action.execute(model)
        logger.info(f"Generation {generation} action executed: {action}")

    return model, combined
