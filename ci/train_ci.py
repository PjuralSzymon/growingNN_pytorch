"""train-ci entry script.

One dataset x one seed. Dataset and seed come from the environment so this
file is not tied to MNIST or any other single algorithm.

User notes:
This is my own new algroithm growingNN follow tests i ahve in tests/regression on how to train it
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

DATASET = os.environ["TRAIN_CI_DATASET"]
SEED = int(os.environ["TRAIN_CI_SEED"])
OUTPUT = Path(os.environ["TRAIN_CI_OUTPUT"])
ROOT = Path(os.environ.get("TRAIN_CI_WORKDIR") or Path(__file__).resolve().parents[1])
TRAINERS = {'cifar10': 'experiments.train_cifar10', 'mnist': 'experiments.train_mnist', 'mnist_lr_schedulers': 'experiments.train_mnist_lr_schedulers', 'mnist_schedulers': 'experiments.train_mnist_schedulers'}
LAUNCH = ''
HAS_EXPERIMENTS_COMMON = True
RESULT_PREFIX = "REGRESSION_CI_RESULT "


def _ensure_repo_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def write_metrics(metric_value: float = 0.0, param_count: int = 0, extra: dict | None = None) -> None:
    payload = {
        "dataset_id": DATASET,
        "dataset": DATASET,
        "seed": SEED,
        "metric_name": "val_acc",
        "metric_value": float(metric_value),
        "param_count": int(param_count),
        "extra": extra or {},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload), encoding="utf-8")
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "dataset": DATASET,
                "seeds": [SEED],
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


def _run_experiment_grid(train_mod: object) -> tuple[float, int] | None:
    if not HAS_EXPERIMENTS_COMMON:
        return None
    common = _import_first(["experiments.experiments_common"])
    if common is None or not hasattr(common, "run_experiment_grid"):
        return None
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slug = DATASET.replace("-", "_")
    data_cls_name = "".join(part.capitalize() for part in slug.split("_")) + "Data"
    data_cls = (
        getattr(train_mod, data_cls_name, None)
        or getattr(train_mod, "MNISTData", None)
        or getattr(train_mod, "Data", None)
    )
    data_dir = getattr(train_mod, "DATA_DIR", ROOT / "data")
    history_name = (
        getattr(train_mod, f"{slug.upper()}_HISTORY_FILENAME", None)
        or getattr(train_mod, "MNIST_HISTORY_FILENAME", None)
        or getattr(train_mod, "HISTORY_FILENAME", "history.pt")
    )
    folder_fn = getattr(train_mod, f"build_{slug}_hyperparameter_folder_name", None) or getattr(
        train_mod, "build_mnist_hyperparameter_folder_name", lambda _hp: DATASET
    )
    model_factory = getattr(train_mod, "MODEL_FACTORY", None)
    if data_cls is None or model_factory is None:
        return None
    data = data_cls(data_dir)
    if hasattr(data, "prepare"):
        data.prepare()
    hp = {"batch_size": 64, "epochs": getattr(train_mod, "EPOCHS", 1)}
    runs_dir = ROOT / "testResults" / "regression" / "ci" / DATASET
    definition = common.ExperimentDefinition(
        name=f"train-ci {DATASET}",
        runs_dir=runs_dir,
        history_filename=history_name,
        seeds=(SEED,),
        folder_name=folder_fn,
        model_factory=model_factory,
        loader_factory=lambda cell: data.loaders(int(cell.get("batch_size") or 64)),
        board_metadata=lambda cell, folder, seed: (f"train-ci {DATASET} seed {seed}", DATASET),
    )
    common.run_experiment_grid(definition, (hp,), device=device, board=False)
    folder = definition.folder_name(hp)
    loaded = _metrics_from_history(runs_dir / folder / f"seed_{SEED}" / history_name)
    return loaded


def run_one() -> tuple[float, int]:
    _ensure_repo_on_path()
    slug = DATASET.replace("-", "_")
    module_name = TRAINERS.get(slug) or TRAINERS.get(DATASET)
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
        via_grid = _run_experiment_grid(train_mod)
        if via_grid is not None:
            return via_grid
        for attr in (f"run_{slug}_regression", "run_regression", "main"):
            fn = getattr(train_mod, attr, None)
            if callable(fn) and attr != "main":
                fn()
                break
    if LAUNCH:
        target = ROOT / LAUNCH
        if target.is_file():
            env = os.environ.copy()
            completed = subprocess.run(
                [sys.executable, str(target)],
                cwd=ROOT,
                env=env,
                check=False,
            )
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
    for history in ROOT.rglob(f"seed_{SEED}"):
        for name in ("history.pt", "mnist_history.pt"):
            loaded = _metrics_from_history(history / name)
            if loaded is not None:
                return loaded
    return 0.0, 0


if __name__ == "__main__":
    acc, params = run_one()
    extra = {"trainer": TRAINERS.get(DATASET.replace("-", "_"), LAUNCH or "")}
    write_metrics(acc, params, extra=extra)
