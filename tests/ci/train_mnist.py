"""train-ci entry script.

One dataset x one seed. Dataset and seed come from the environment so this
file is not tied to MNIST or any other single algorithm.

MNIST uses the Exp 005 keep-set package on the Exp 004 LR cell:
- Exp 001: slope gate 3°, logistic recovery
- Exp 003 after_fix: val_acc grading, big starter
- Exp 004: composed_exponential LR
- Exp 005: sequential_halving_beam search (top accuracy and composite; not MCTS)

CI length is 8 generations x 8 epochs (64 train epochs). Exp 005 used 10 x 10.
This is a shorter gate; some Exp 005 seeds only crossed val 0.85 after generation 8.

This is a gate, not an experiment.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(os.environ.get("TRAIN_CI_WORKDIR") or Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    INITIAL_LR,
    MODEL_FACTORY,
    MODEL_NAME,
    SCORE_ACCURACY_METRIC,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    build_learning_rate_scheduler_for_schedule_id,
)
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler

TRAINERS = {
    "cifar10": "experiments.train_cifar10",
    "mnist": "experiments.train_mnist",
    "mnist_lr_schedulers": "experiments.train_mnist_lr_schedulers",
    "mnist_schedulers": "experiments.train_mnist_schedulers",
}
LAUNCH = ""
HAS_EXPERIMENTS_COMMON = True
RESULT_PREFIX = "REGRESSION_CI_RESULT "
SCHEDULE_ID = "composed_exponential"
SIMULATION_ALG_ID = "sequential_halving_beam"
SIMULATION_ALG = sequential_halving_beam_alg
EPOCHS_PER_GENERATION = 10
GENERATIONS = 8


def _ci_env() -> tuple[str, int, Path]:
    return (
        os.environ["TRAIN_CI_DATASET"],
        int(os.environ["TRAIN_CI_SEED"]),
        Path(os.environ["TRAIN_CI_OUTPUT"]),
    )


def write_metrics(
    metric_value: float = 0.0,
    param_count: int = 0,
    extra: dict | None = None,
    *,
    dataset: str,
    seed: int,
    output: Path,
) -> None:
    payload = {
        "dataset_id": dataset,
        "dataset": dataset,
        "seed": seed,
        "metric_name": "val_acc",
        "metric_value": float(metric_value),
        "param_count": int(param_count),
        "extra": extra or {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "dataset": dataset,
                "seeds": [seed],
                "val_acc": [float(metric_value)],
                "param_count": [int(param_count)],
            }
        ),
        flush=True,
    )


def _import_first(names: list[str]):
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _metrics_from_history(history_path: Path) -> tuple[float, int] | None:
    if not history_path.is_file():
        return None
    try:
        import torch
    except ImportError:
        return None
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    if not isinstance(history, dict) or "val_acc" not in history:
        return None
    acc = float(history["val_acc"][-1])
    params = history.get("param_count")
    count = int(params[-1]) if isinstance(params, list) and params else int(params or 0)
    return acc, count


def mnist_hyperparameters() -> dict[str, object]:
    """Return the Exp 005 sequential_halving_beam cell on the Exp 004 LR package."""
    values = next(itertools.product(*train_mnist.METAPARAM_LISTS))
    return {
        **dict(zip(train_mnist.METAPARAM_KEYS, values)),
        "epochs": EPOCHS_PER_GENERATION,
        "generations": GENERATIONS,
        "simulation_time": SIMULATION_TIME_SEC,
        "lr_alpha": INITIAL_LR,
        "score_accuracy_metric": SCORE_ACCURACY_METRIC,
        "simulation_alg_id": SIMULATION_ALG_ID,
        "simulation_alg": SIMULATION_ALG,
        "model_name": MODEL_NAME,
        "lr_scheduler_factory": (
            lambda hp: build_learning_rate_scheduler_for_schedule_id(SCHEDULE_ID, hp)
        ),
    }


def run_mnist(*, seed: int, root: Path) -> tuple[float, int]:
    """Train one MNIST seed with Exp 005 sequential_halving_beam on Exp 004 LR."""
    configure_deterministic_seeding()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()
    hp = mnist_hyperparameters()
    runs_dir = root / "testResults" / "regression" / "ci" / "mnist"
    definition = common.ExperimentDefinition(
        name=f"train-ci mnist {SCHEDULE_ID} {MODEL_NAME}",
        runs_dir=runs_dir,
        history_filename=train_mnist.MNIST_HISTORY_FILENAME,
        seeds=(seed,),
        folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
        model_factory=MODEL_FACTORY,
        loader_factory=lambda cell: data.loaders(int(cell["batch_size"])),
        board_metadata=lambda cell, folder, run_seed: (
            f"train-ci mnist {SIMULATION_ALG_ID} {SCHEDULE_ID} {MODEL_NAME} | {folder} | seed {run_seed}",
            "MNIST",
        ),
    )
    with patch.object(
        common,
        "AlwaysSimulationScheduler",
        partial(
            SlopeEstimationSimulationScheduler,
            angle_threshold=SLOPE_ANGLE_THRESHOLD,
        ),
    ):
        common.run_experiment_grid(definition, (hp,), device=device, board=False)
    folder = definition.folder_name(hp)
    loaded = _metrics_from_history(
        runs_dir / folder / f"seed_{seed}" / train_mnist.MNIST_HISTORY_FILENAME
    )
    if loaded is None:
        raise RuntimeError(f"MNIST CI run did not write history under {runs_dir / folder / f'seed_{seed}'}")
    return loaded


def _run_experiment_grid(train_mod: object, *, dataset: str, seed: int, root: Path) -> tuple[float, int] | None:
    if not HAS_EXPERIMENTS_COMMON or not hasattr(common, "run_experiment_grid"):
        return None
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slug = dataset.replace("-", "_")
    data_cls_name = "".join(part.capitalize() for part in slug.split("_")) + "Data"
    data_cls = (
        getattr(train_mod, data_cls_name, None)
        or getattr(train_mod, "MNISTData", None)
        or getattr(train_mod, "Data", None)
    )
    data_dir = getattr(train_mod, "DATA_DIR", root / "data")
    history_name = (
        getattr(train_mod, f"{slug.upper()}_HISTORY_FILENAME", None)
        or getattr(train_mod, "MNIST_HISTORY_FILENAME", None)
        or getattr(train_mod, "HISTORY_FILENAME", "history.pt")
    )
    folder_fn = getattr(train_mod, f"build_{slug}_hyperparameter_folder_name", None) or getattr(
        train_mod, "build_mnist_hyperparameter_folder_name", lambda _hp: dataset
    )
    model_factory = getattr(train_mod, "MODEL_FACTORY", None) or getattr(
        train_mod, "_build_model", None
    )
    if data_cls is None or model_factory is None:
        return None
    data = data_cls(data_dir)
    if hasattr(data, "prepare"):
        data.prepare()
    epochs = getattr(train_mod, "EPOCHS", 1)
    if isinstance(epochs, (list, tuple)):
        epochs = epochs[0]
    hp = {"batch_size": 64, "epochs": epochs}
    runs_dir = root / "testResults" / "regression" / "ci" / dataset
    definition = common.ExperimentDefinition(
        name=f"train-ci {dataset}",
        runs_dir=runs_dir,
        history_filename=history_name,
        seeds=(seed,),
        folder_name=folder_fn,
        model_factory=model_factory,
        loader_factory=lambda cell: data.loaders(int(cell.get("batch_size") or 64)),
        board_metadata=lambda cell, folder, run_seed: (f"train-ci {dataset} seed {run_seed}", dataset),
    )
    common.run_experiment_grid(definition, (hp,), device=device, board=False)
    folder = definition.folder_name(hp)
    return _metrics_from_history(runs_dir / folder / f"seed_{seed}" / history_name)


def run_one(*, dataset: str, seed: int, root: Path) -> tuple[float, int]:
    slug = dataset.replace("-", "_")
    if slug == "mnist":
        return run_mnist(seed=seed, root=root)
    module_name = TRAINERS.get(slug) or TRAINERS.get(dataset)
    candidates = [module_name] if module_name else []
    candidates.extend(
        [
            f"experiments.train_{slug}",
            f"train_{slug}",
            "experiments.train",
            "train",
        ]
    )
    train_mod = _import_first([name for name in candidates if name])
    if train_mod is not None:
        via_grid = _run_experiment_grid(train_mod, dataset=dataset, seed=seed, root=root)
        if via_grid is not None:
            return via_grid
        for attr in (f"run_{slug}_regression", "run_regression"):
            fn = getattr(train_mod, attr, None)
            if callable(fn):
                fn()
                break
    if LAUNCH:
        target = root / LAUNCH
        if target.is_file():
            completed = subprocess.run(
                [sys.executable, str(target)],
                cwd=root,
                env=os.environ.copy(),
                check=False,
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
    for history in root.rglob(f"seed_{seed}"):
        for name in ("history.pt", "mnist_history.pt", "train_mnist_history.pt"):
            loaded = _metrics_from_history(history / name)
            if loaded is not None:
                return loaded
    raise RuntimeError(f"train-ci produced no metrics for dataset={dataset!r} seed={seed}")


if __name__ == "__main__":
    dataset, seed, output = _ci_env()
    acc, params = run_one(dataset=dataset, seed=seed, root=ROOT)
    extra = {
        "trainer": TRAINERS.get(dataset.replace("-", "_"), LAUNCH or ""),
        "schedule_id": SCHEDULE_ID if dataset.replace("-", "_") == "mnist" else "",
        "simulation_alg_id": SIMULATION_ALG_ID if dataset.replace("-", "_") == "mnist" else "",
        "model": MODEL_NAME if dataset.replace("-", "_") == "mnist" else "",
    }
    write_metrics(acc, params, extra=extra, dataset=dataset, seed=seed, output=output)
