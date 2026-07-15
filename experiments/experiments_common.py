"""Shared execution lifecycle for growingNN experiment scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader

from growingnn.board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_scheduler import SchedulerMode, SimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph

METRIC_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
Hyperparameters = dict[str, object]
LoaderSet = tuple[DataLoader, DataLoader, DataLoader]
BoardMetadata = tuple[str, str]


@dataclass(frozen=True)
class ExperimentDefinition:
    """Experiment-specific callbacks and output settings used by the shared runner."""

    name: str
    runs_dir: Path
    history_filename: str
    seeds: Sequence[int]
    folder_name: Callable[[Hyperparameters], str]
    model_factory: Callable[[Hyperparameters], nn.Module]
    loader_factory: Callable[[Hyperparameters], LoaderSet]
    board_metadata: Callable[[Hyperparameters, str, int], BoardMetadata]
    save_fx_graphs: bool = False


def parse_board_cli(description: str) -> argparse.Namespace:
    """Parse the board output switch shared by experiment scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--board",
        choices=("true", "false"),
        default="true",
        help="Write GrowingNN Board artifacts under each run's board/ folder (default: true)",
    )
    args = parser.parse_args()
    args.board = args.board == "true"
    return args


def require_cuda(device: torch.device) -> None:
    """Validate that the configured CUDA device can execute a small convolution."""
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; torch.cuda.is_available() is False")
    cap = torch.cuda.get_device_capability(device)
    logger.info(
        "Training device: %s (%s, sm_%d%d, torch %s)",
        device, torch.cuda.get_device_name(device), cap[0], cap[1], torch.__version__,
    )
    try:
        torch.nn.Conv2d(3, 8, 3).to(device)(
            torch.zeros(1, 3, 32, 32, device=device)
        )
    except RuntimeError as exc:
        if "no kernel image" in str(exc).lower():
            arch = getattr(torch.cuda, "get_arch_list", lambda: [])()
            raise RuntimeError(
                f"PyTorch {torch.__version__} has no CUDA kernels for "
                f"{torch.cuda.get_device_name(device)} (sm_{cap[0]}{cap[1]}). "
                f"Supported arches: {arch or 'unknown'}. RTX 50-series often needs cu128 wheels: "
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
            ) from exc
        raise


def _running_config(
    hp: Hyperparameters,
    device: torch.device,
    board: ExperimentBoard | None,
) -> RunningConfig:
    return RunningConfig(
        generations=int(hp["generations"]),
        epochs=int(hp["epochs"]),
        device=device,
        lr_scheduler=LearningRateScheduler(
            ScheduleMode.PROGRESSIVE_PARABOLIC, alpha=float(hp["lr_alpha"])
        ),
        simulation_alg=montecarlo_alg,
        simulation_scheduler=SimulationScheduler(
            SchedulerMode.ALWAYS,
            simulation_time=float(hp["simulation_time"]),
            simulation_epochs=int(hp["simulation_epochs"]),
        ),
        stopper=AccuracyStopper(target_accuracy=float(hp["target_accuracy"])),
        simulation_score=SimulationScore(
            weight_acc=float(hp["score_weight_acc"]),
            weight_countW=float(hp["score_weight_countw"]),
        ),
        simulation_set_size=int(hp["simulation_set_size"]),
        criterion=nn.CrossEntropyLoss(),
        quiet=False,
        enable_experiment_board=board is not None,
        experiment_board=board,
    )


def _draw_graphs(run_dir: Path, suffix: str, gm: fx.GraphModule) -> None:
    simplified_name = (
        "fx_graph_error_simplified"
        if suffix == "_error"
        else f"fx_graph_simplified{suffix}"
    )
    draw_filtered_fx_graph(gm, str(run_dir / simplified_name), fmt="pdf")
    draw_torch_fx_graph(gm, str(run_dir / f"fx_graph{suffix}"), fmt="pdf")


def _save_artifacts(
    run_dir: Path,
    history_filename: str,
    summary: dict[str, list[Any]],
) -> None:
    history = {key: summary[key] for key in METRIC_KEYS}
    torch.save(history, run_dir / history_filename)
    for name, values in history.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(values) + 1), values)
        ax.set_xlabel("step")
        ax.set_ylabel(name)
        fig.tight_layout()
        fig.savefig(run_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def _train_run(
    definition: ExperimentDefinition,
    hp: Hyperparameters,
    *,
    folder: str,
    seed: int,
    run_dir: Path,
    device: torch.device,
    board_enabled: bool,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gm = fx.symbolic_trace(definition.model_factory(hp))
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)
    logger.info("Run %s seed %s params=%s -> %s", folder, seed, params_before, run_dir)
    if definition.save_fx_graphs:
        _draw_graphs(run_dir, "0", gm)

    experiment_name, dataset = definition.board_metadata(hp, folder, seed)
    board = (
        ExperimentBoard(
            run_dir / "board",
            experiment_name=experiment_name,
            dataset=dataset,
            device=str(device),
        )
        if board_enabled
        else None
    )
    config = _running_config(hp, device, board)
    train_loader, val_loader, clean_train_loader = definition.loader_factory(hp)
    sim_train, sim_val = sample_loaders(
        clean_train_loader,
        val_loader,
        int(hp["simulation_set_size"]),
        seed=seed,
    )

    try:
        trained_model, summary = train_generations(
            gm,
            train_loader,
            val_loader,
            config,
            sim_train_loader=sim_train,
            sim_val_loader=sim_val,
        )
    except Exception:
        if definition.save_fx_graphs:
            _draw_graphs(run_dir, "_error", gm)
        raise

    if definition.save_fx_graphs:
        _draw_graphs(run_dir, str(summary["generation"][-1]), trained_model)
    _save_artifacts(run_dir, definition.history_filename, summary)
    params_after = GraphStructureQuery.get_amount_of_parameters(trained_model)
    logger.info(
        "Done %s seed %s val_acc=%.4f params %s -> %s changed=%s",
        folder, seed, max(summary["val_acc"]), params_before, params_after,
        params_after != params_before,
    )


def run_experiment_grid(
    definition: ExperimentDefinition,
    hyperparameters: Iterable[Hyperparameters],
    *,
    device: torch.device,
    board: bool,
) -> tuple[int, int]:
    """Run missing grid entries and return executed and skipped run counts."""
    definition.runs_dir.mkdir(parents=True, exist_ok=True)
    executed = 0
    skipped = 0
    for hp in hyperparameters:
        folder = definition.folder_name(hp)
        for seed in definition.seeds:
            run_dir = definition.runs_dir / folder / f"seed_{seed}"
            if run_dir.exists():
                logger.info("Skipping existing %s seed %s", folder, seed)
                skipped += 1
                continue
            run_dir.mkdir(parents=True)
            _train_run(
                definition,
                hp,
                folder=folder,
                seed=seed,
                run_dir=run_dir,
                device=device,
                board_enabled=board,
            )
            executed += 1
    logger.info("%s grid finished: executed=%s skipped=%s", definition.name, executed, skipped)
    return executed, skipped
