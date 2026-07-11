"""Generation loop: train, optionally simulate an architecture mutation, repeat."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.gradient_descent import gradient_descent
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def train_generations(
    model: nn.Module | fx.GraphModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: RunningConfig,
    *,
    sim_train_loader: DataLoader | None = None,
    sim_val_loader: DataLoader | None = None,
) -> tuple[nn.Module | fx.GraphModule, dict[str, list[Any]]]:
    logger.info("Training generations started")

    board = config.experiment_board
    if board is not None:
        board.on_run_start(model, config)

    if sim_train_loader is None or sim_val_loader is None:
        sim_train_loader, sim_val_loader = sample_loaders(
            train_loader, val_loader, config.simulation_set_size
        )
    config.set_simulation_loaders(sim_train_loader, sim_val_loader)

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

    for generation in range(config.generations):
        logger.info("Training generation %s started", generation)
        if board is not None:
            board.on_generation_start(generation, model)

        model, history = gradient_descent(
            model,
            config.epochs,
            train_loader,
            val_loader,
            config.criterion,
            config.lr_scheduler,
            stopper=config.stopper,
            quiet=config.quiet,
            print_every=config.print_every,
            experiment_board=board,
            generation=generation,
            device=config.device,
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

        if board is not None:
            board.on_generation_end(generation, model, history, param_count)

        metrics = {"accuracy": history["train_acc"][-1], "val_acc": val_acc}
        if config.stopper.check(model, generation, metrics):
            break

        if config.simulation_scheduler.can_simulate(generation, generation_val_acc, quiet=config.quiet):
            action, _, _ = config.simulation_alg.get_action(copy.deepcopy(model), config)
            if action is None:
                logger.warning("Generation %s no action executed", generation)
                continue
            action.execute(model)
            logger.info("Generation %s action executed: %s", generation, action)
            if board is not None:
                board.save_graphs(model, generation=generation, tag=f"gen_{generation}_simulation")
                board.note_simulation_graph_saved(generation)

    clear_reshepers_cache()
    if board is not None:
        board.on_run_finished()
    return model, combined
