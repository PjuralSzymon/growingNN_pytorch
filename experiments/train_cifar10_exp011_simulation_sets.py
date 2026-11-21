"""Experiment 011: compare simulation-set selectors under the Experiment 010 MCTS settings.

The search settings stay at the Pass 2 winners. Only the rule that builds the
10% simulation set changes. The selectors are stratified random, EL2N,
coverage-centric selection, and moderate. Datasets are CIFAR-10, MNIST, and
CIFAR-100. Finished CIFAR-10 folders stay at the original path and are reused.

    pipenv run python -u experiments/train_cifar10_exp011_simulation_sets.py
    pipenv run python -u experiments/train_cifar10_exp011_simulation_sets.py --dataset mnist --algorithm moderate --seed 1

Completed run folders are reused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config as project_config
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_cifar10_exp009_best_reference import get_datasets
from growingnn.board.experiment_board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_sets import (
    CcsSimulationSet,
    El2nSimulationSet,
    ModerateSimulationSet,
    SimulationSetAlgorithm,
    StratifiedRandomSimulationSet,
)
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

GENERATIONS = 10
EPOCHS = 20
SEEDS = (1, 2)
DATASETS = ("cifar10", "mnist", "cifar100")
SIMULATION_PERCENT = 10.0
SIMULATION_TIME_SECONDS = 32.0
RUNS_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp011_simulation_sets"


class TinyImageStarter(nn.Module):
    """One 8-channel convolution and a linear head. Channel count and classes follow the dataset."""

    def __init__(self, in_channels: int, classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self.linear2 = nn.Linear(8, classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


def initial_graph(dataset: str) -> nn.Module:
    """Build the tiny starter for CIFAR-10, MNIST, or CIFAR-100."""
    if dataset == "mnist":
        return TinyImageStarter(1, 10)
    if dataset == "cifar10":
        return TinyImageStarter(3, 10)
    if dataset == "cifar100":
        return TinyImageStarter(3, 100)
    raise ValueError(f"Experiment 011 does not support dataset {dataset!r}")

# Pass 2 winners from Experiment 010. Simulation-set size is fixed at 10% for every arm.
PASS2_MCTS = {
    "MCTS_UCB1_C": 1.0,
    "MCTS_UCB1_USE_SQRT": False,
    "MCTS_PROPAGATE_ROLLOUT_VALUE": True,
    "MCTS_ROLLOUT_DEPTH": 1,
    "MCTS_ROLLOUT_EPOCHS": 1,
    "MCTS_ROLLOUT_LR": 0.0001,
    "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS": 3,
}
SELECTORS: tuple[tuple[str, type[SimulationSetAlgorithm]], ...] = (
    ("stratified_random", StratifiedRandomSimulationSet),
    ("el2n", El2nSimulationSet),
    ("ccs", CcsSimulationSet),
    ("moderate", ModerateSimulationSet),
)
ALGORITHM_NAMES = tuple(name for name, _selector in SELECTORS)


def requested_simulation_size(clean_count: int, percent: float = SIMULATION_PERCENT) -> int:
    """Convert a percentage of the clean training split into a sample count."""
    return max(1, min(clean_count, round(clean_count * percent / 100.0)))


def build_selector(name: str) -> SimulationSetAlgorithm:
    """Return one fresh selector for an algorithm name."""
    for algorithm_name, selector_type in SELECTORS:
        if algorithm_name == name:
            return selector_type()
    raise ValueError(f"Unknown simulation-set algorithm {name!r}. Expected one of: {', '.join(ALGORITHM_NAMES)}")


def apply_mcts_settings() -> None:
    """Install the Experiment 010 Pass 2 MCTS settings on the process globals."""
    for name, value in PASS2_MCTS.items():
        if name == "MCTS_ROLLOUT_LR":
            value = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=float(value))
        setattr(project_config, name, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read which algorithms and seeds to run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", action="append", dest="algorithms", choices=ALGORITHM_NAMES)
    parser.add_argument("--dataset", action="append", dest="datasets", choices=DATASETS)
    parser.add_argument("--seed", action="append", dest="seeds", type=int)
    return parser.parse_args(argv)


def selected_datasets(names: list[str] | None) -> tuple[str, ...]:
    """Use CIFAR-10, MNIST, and CIFAR-100 when the command line does not name one."""
    if not names:
        return DATASETS
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Experiment 011 datasets must be in {DATASETS}, got {unknown}")
    return tuple(names)


def selected_algorithms(names: list[str] | None) -> tuple[str, ...]:
    """Use every selector when the command line does not name one."""
    if not names:
        return ALGORITHM_NAMES
    return tuple(names)


def selected_seeds(seeds: list[int] | None) -> tuple[int, ...]:
    """Keep the comparison on the Experiment 010 seeds."""
    if not seeds:
        return SEEDS
    unknown = [seed for seed in seeds if seed not in SEEDS]
    if unknown:
        raise ValueError(f"Experiment 011 seeds must be in {SEEDS}, got {unknown}")
    return tuple(seeds)


def summarize_results(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank algorithms by mean final validation accuracy inside each dataset."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        key = (str(run["dataset"]), str(run["algorithm"]))
        grouped.setdefault(key, []).append(float(run["final_validation_accuracy"]))
    summary = [
        {
            "dataset": dataset,
            "algorithm": algorithm,
            "runs": len(values),
            "mean_final_validation_accuracy": sum(values) / len(values),
        }
        for (dataset, algorithm), values in grouped.items()
    ]
    return sorted(summary, key=lambda row: (row["dataset"], -row["mean_final_validation_accuracy"]))


def run_directory(dataset: str, algorithm: str, seed: int) -> Path:
    """Return the folder for one run. CIFAR-10 keeps the path used by the finished grid."""
    if dataset == "cifar10":
        return RUNS_DIR / algorithm / f"seed_{seed}"
    return RUNS_DIR / dataset / algorithm / f"seed_{seed}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_one(algorithm: str, seed: int, data: Any, output_dir: Path, dataset: str = "cifar10") -> dict[str, Any]:
    """Train one selector, dataset, and seed, or return the saved result when that run already finished."""
    history_path = output_dir / "history.pt"
    metadata_path = output_dir / "run.json"
    if history_path.is_file() and metadata_path.is_file():
        print(f"SKIP {dataset} {algorithm} seed {seed} history exists at {output_dir}", flush=True)
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    apply_mcts_settings()
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initial_graph(dataset)
    train_loader, val_loader, clean_loader = data.loaders(128)
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(128)
    requested_size = requested_simulation_size(len(clean_loader.dataset))
    sim_train, sim_val = build_selector(algorithm).sample_loaders(
        clean_loader,
        val_loader,
        requested_size,
        seed=seed,
    )
    board = ExperimentBoard(
        output_dir / "board",
        experiment_id=f"exp011_{algorithm}_{dataset}_seed{seed}",
        experiment_name="Experiment 011 simulation-set comparison",
        dataset=dataset,
        device=str(device),
    )
    config = RunningConfig(
        generations=GENERATIONS,
        epochs=EPOCHS,
        lr_scheduler=CosineDecayScheduler(total_epochs=GENERATIONS * EPOCHS),
        stopper=AccuracyStopper(target_accuracy=0.999),
        simulation_alg=montecarlo_alg,
        simulation_scheduler=AlwaysSimulationScheduler(SIMULATION_TIME_SECONDS, 1),
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
        simulation_set_size=len(sim_train.dataset),
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        device=device,
        momentum=0.9,
        weight_decay=0.0001,
        max_grad_norm=1.0,
        seq_conv_insert_before_global_pool=True,
        enable_experiment_board=True,
        experiment_board=board,
    )
    print(
        f"EXP011 dataset={dataset} algorithm={algorithm} seed={seed} sim_train={len(sim_train.dataset)}/{len(clean_loader.dataset)} "
        f"sim_val={len(sim_val.dataset)} time={SIMULATION_TIME_SECONDS}s",
        flush=True,
    )
    _grown, summary = train_generations(
        model,
        train_loader,
        val_loader,
        config,
        sim_train_loader=sim_train,
        sim_val_loader=sim_val,
        test_loader=test_loader,
    )
    history = {
        "experiment": "011_simulation_sets",
        "algorithm": algorithm,
        "dataset": dataset,
        "seed": seed,
        "generations": GENERATIONS,
        "epochs_per_generation": EPOCHS,
        "simulation_percent": SIMULATION_PERCENT,
        "simulation_train_size_requested": requested_size,
        "simulation_train_size_actual": len(sim_train.dataset),
        "simulation_val_size_actual": len(sim_val.dataset),
        "simulation_train_indices": [int(index) for index in sim_train.dataset.indices],
        "simulation_val_indices": [int(index) for index in sim_val.dataset.indices],
        "mcts_settings": dict(PASS2_MCTS),
        **{key: list(summary[key]) for key in summary},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(history, history_path)
    result = {
        "algorithm": algorithm,
        "dataset": dataset,
        "seed": seed,
        "final_train_accuracy": float(summary["train_acc"][-1]),
        "final_validation_accuracy": float(summary["val_acc"][-1]),
        "best_validation_accuracy": float(max(summary["val_acc"])),
        "final_test_accuracy": float(summary["test_acc"][-1]),
        "simulation_train_size_actual": len(sim_train.dataset),
        "simulation_val_size_actual": len(sim_val.dataset),
        "history": str(history_path),
    }
    _write_json(metadata_path, result)
    print(f"RESULT {json.dumps(result)}", flush=True)
    return result


def run_experiment(
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
    datasets: tuple[str, ...] = DATASETS,
) -> list[dict[str, Any]]:
    """Run the requested grid and write the ranking by mean final validation accuracy."""
    for dataset in datasets:
        data = get_datasets(dataset)
        for algorithm in algorithms:
            for seed in seeds:
                run_one(algorithm, seed, data, run_directory(dataset, algorithm, seed), dataset)
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RUNS_DIR.glob("**/run.json"))
    ]
    summary = summarize_results(runs)
    _write_json(RUNS_DIR / "summary.json", {"runs": runs, "ranking": summary})
    print(f"SUMMARY {json.dumps(summary)}", flush=True)
    return runs


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(
        selected_algorithms(args.algorithms),
        selected_seeds(args.seeds),
        selected_datasets(args.datasets),
    )


if __name__ == "__main__":
    main()
