"""Generation loop: train, optionally simulate an architecture mutation, repeat."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.core.logger import logger
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.gradient_descent import _evaluate, gradient_descent
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def train_generations(
    model: nn.Module | fx.GraphModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: RunningConfig,
    *,
    sim_train_loader: DataLoader | None = None,
    sim_val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
) -> tuple[nn.Module | fx.GraphModule, dict[str, list[Any]]]:
    logger.info("Training generations started")

    inputs, _ = next(iter(train_loader))
    traced = TracedModel.create(model, tuple(int(x) for x in inputs[0:1].shape))

    board = config.experiment_board
    if board is not None:
        board.on_run_start(traced.gm, config)

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
        "action_generation": [],
        "action_name": [],
        "action_from": [],
        "action_to": [],
        "action_val_acc_before": [],
        "action_val_acc_after": [],
    }
    if test_loader is not None:
        combined["test_loss"] = []
        combined["test_acc"] = []

    for generation in range(config.generations):
        logger.info("Training generation %s started", generation)
        if board is not None:
            board.on_generation_start(generation, traced.gm)

        traced.gm, history = gradient_descent(
            traced.gm,
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
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
        )
        val_acc = history["val_acc"][-1]
        generation_val_acc.append(val_acc)
        combined["generation"].append(generation)
        combined["train_loss"].extend(history["train_loss"])
        combined["train_acc"].extend(history["train_acc"])
        combined["val_loss"].extend(history["val_loss"])
        combined["val_acc"].extend(history["val_acc"])
        combined["lr"].extend(history["lr"])
        param_count = traced.param_count()
        combined["param_count"].extend([param_count] * len(history["train_loss"]))
        if test_loader is not None:
            test_loss, test_acc = _evaluate(
                traced.gm,
                test_loader,
                config.criterion,
                torch.device(config.device),
            )
            combined["test_loss"].append(test_loss)
            combined["test_acc"].append(test_acc)
            logger.info(
                "Generation %s test_acc %.4f test_loss %.4f",
                generation, test_acc, test_loss,
            )

        if board is not None:
            board.on_generation_end(generation, traced.gm, history, param_count)

        metrics = {"accuracy": history["train_acc"][-1], "val_acc": val_acc}
        if config.stopper.check(traced.gm, generation, metrics):
            break

        # Slope estimation uses start/middle/end training accuracy from this generation;
        # other schedulers keep using final accuracies across generations.
        scheduler_values = (
            history["train_acc"]
            if config.simulation_scheduler.uses_current_generation_values
            else generation_val_acc
        )
        if config.simulation_scheduler.can_simulate(
            generation,
            scheduler_values,
            quiet=config.quiet,
            generations=config.generations,
        ):
            action, _, _ = config.simulation_alg.get_action(copy.deepcopy(traced), config)
            if action is None:
                logger.warning("Generation %s no action executed", generation)
                continue
            eval_device = torch.device(config.device)
            pre_acc = None
            post_acc = None
            if config.criterion is not None:
                _, pre_acc = _evaluate(traced.gm, val_loader, config.criterion, eval_device)
            action.execute(traced)
            if config.criterion is not None:
                _, post_acc = _evaluate(traced.gm, val_loader, config.criterion, eval_device)
            config.lr_scheduler.structure_changed()
            params = list(getattr(action, "params", []) or [])
            action_from = params[0] if params else ""
            action_to = params[1] if len(params) > 1 else ""
            combined["action_generation"].append(generation)
            combined["action_name"].append(type(action).__name__)
            combined["action_from"].append(str(action_from))
            combined["action_to"].append(str(action_to))
            combined["action_val_acc_before"].append(pre_acc)
            combined["action_val_acc_after"].append(post_acc)
            if pre_acc is None or post_acc is None:
                logger.info(
                    "Generation %s action executed: %s %s -> %s",
                    generation, type(action).__name__, action_from, action_to,
                )
            else:
                logger.info(
                    "Generation %s action %s %s -> %s val_acc %.4f -> %.4f",
                    generation, type(action).__name__, action_from, action_to, pre_acc, post_acc,
                )
            if board is not None:
                board.save_graphs(traced.gm, generation=generation, tag=f"gen_{generation}_simulation")
                board.note_simulation_graph_saved(generation)

    clear_reshepers_cache()
    if board is not None:
        board.on_run_finished()
    return traced.gm, combined
