"""train-ci entry script.

One dataset x one seed. Dataset and seed come from the environment so this
file is not tied to MNIST or any other single algorithm.

Follow tests/regression (especially tests/regression/ci) and experiment
helpers (train_*_exp*) for how to train. This is a gate, not an experiment.

User notes:
(none)
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("TRAIN_CI_WORKDIR") or Path(__file__).resolve().parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAINERS = {'mnist': 'experiments.train_mnist'}
HELPERS = {'mnist': ['experiments.train_mnist_exp001_slope_model_depth', 'experiments.train_mnist_exp002_initial_architectures', 'experiments.train_mnist_exp003_score_accuracy_metric', 'experiments.train_mnist_exp004_composed_lr_schedulers']}
REGRESSION = {'adding_neurons': 'tests.regression.actions.adding_neurons', 'adding_res_conv_layers': 'tests.regression.actions.adding_res_conv_layers', 'adding_res_layers': 'tests.regression.actions.adding_res_layers', 'adding_seq_conv_layers': 'tests.regression.actions.adding_seq_conv_layers', 'adding_seq_layers': 'tests.regression.actions.adding_seq_layers', 'big_models_all_action_test': 'tests.regression.actions.big_models_all_action_test', 'del_layers': 'tests.regression.actions.del_layers', 'del_neurons': 'tests.regression.actions.del_neurons', 'resnet_all_action_test': 'tests.regression.actions.resnet_all_action_test', 'resnet_regression_test': 'tests.regression.actions.resnet_regression_test', 'regression_utils': 'tests.regression.regression_utils', 'simple_gradient_descent': 'tests.regression.training.simple_gradient_descent', 'trainer_generations': 'tests.regression.training.trainer_generations'}
LAUNCH = ''
HAS_EXPERIMENTS_COMMON = True
PREFERRED_SCHEDULE_ID = 'composed_step'
TARGET_DATASETS = ('mnist',)
RESULT_PREFIX = "REGRESSION_CI_RESULT "
HISTORY_NAMES = ("history.pt", "mnist_history.pt", "train_mnist_history.pt")


def _ci_env() -> tuple[str, int, Path]:
    return (
        os.environ["TRAIN_CI_DATASET"],
        int(os.environ["TRAIN_CI_SEED"]),
        Path(os.environ["TRAIN_CI_OUTPUT"]),
    )


def write_metrics(
    metric_value: float,
    param_count: int,
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


def _import_first(names: list[str | None]):
    for name in names:
        if not name:
            continue
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


def _load_helpers(slug: str) -> list[object]:
    loaded: list[object] = []
    for name in HELPERS.get(slug) or []:
        mod = _import_first([name])
        if mod is not None:
            loaded.append(mod)
    return loaded


def _first_attr(mods: list[object], *names: str):
    for mod in mods:
        for name in names:
            if hasattr(mod, name):
                return getattr(mod, name)
    return None


def _hyperparameters(train_mod: object, helpers: list[object]) -> dict[str, object]:
    hp: dict[str, object] = {}
    if hasattr(train_mod, "METAPARAM_LISTS") and hasattr(train_mod, "METAPARAM_KEYS"):
        values = next(itertools.product(*train_mod.METAPARAM_LISTS))
        hp.update(dict(zip(train_mod.METAPARAM_KEYS, values)))
    sources = [*helpers, train_mod]
    epochs = _first_attr(sources, "EPOCHS_PER_GENERATION", "EPOCHS")
    if isinstance(epochs, (list, tuple)):
        epochs = epochs[0]
    hp.setdefault("epochs", epochs if epochs is not None else 1)
    hp.setdefault("batch_size", 64)
    generations = _first_attr(sources, "GENERATIONS")
    if generations is not None:
        hp["generations"] = generations
    simulation_time = _first_attr(sources, "SIMULATION_TIME_SEC")
    if simulation_time is not None:
        hp["simulation_time"] = simulation_time
    lr_alpha = _first_attr(sources, "INITIAL_LR")
    if lr_alpha is not None:
        hp["lr_alpha"] = lr_alpha
    metric = _first_attr(sources, "SCORE_ACCURACY_METRIC")
    if metric is not None:
        hp["score_accuracy_metric"] = metric
    builder = _first_attr(sources, "build_learning_rate_scheduler_for_schedule_id")
    if callable(builder):
        schedule_id = PREFERRED_SCHEDULE_ID or "composed_step"
        hp["lr_scheduler_factory"] = lambda cell, _b=builder, _sid=schedule_id: _b(_sid, cell)
    return hp


def _configure_seeding(helpers: list[object]) -> None:
    fn = _first_attr(helpers, "configure_deterministic_seeding")
    if callable(fn):
        fn()


def _scheduler_patch(common: object, helpers: list[object]):
    threshold = _first_attr(helpers, "SLOPE_ANGLE_THRESHOLD")
    if threshold is None or not hasattr(common, "AlwaysSimulationScheduler"):
        return contextlib.nullcontext()
    try:
        from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
    except ImportError:
        return contextlib.nullcontext()
    return patch.object(
        common,
        "AlwaysSimulationScheduler",
        partial(SlopeEstimationSimulationScheduler, angle_threshold=threshold),
    )


def _model_factory(train_mod: object, helpers: list[object]):
    return _first_attr([*helpers, train_mod], "MODEL_FACTORY", "_build_model")


def _run_experiment_grid(
    train_mod: object,
    helpers: list[object],
    *,
    dataset: str,
    seed: int,
    root: Path,
) -> tuple[float, int] | None:
    if not HAS_EXPERIMENTS_COMMON:
        return None
    common = _import_first(["experiments.experiments_common"])
    if common is None or not hasattr(common, "run_experiment_grid"):
        return None
    import torch

    _configure_seeding(helpers)
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
    model_factory = _model_factory(train_mod, helpers)
    if data_cls is None or model_factory is None:
        return None
    data = data_cls(data_dir)
    if hasattr(data, "prepare"):
        data.prepare()
    hp = _hyperparameters(train_mod, helpers)
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
    with _scheduler_patch(common, helpers):
        common.run_experiment_grid(definition, (hp,), device=device, board=False)
    folder = definition.folder_name(hp)
    return _metrics_from_history(runs_dir / folder / f"seed_{seed}" / history_name)


def run_one(*, dataset: str, seed: int, root: Path) -> tuple[float, int]:
    slug = dataset.replace("-", "_")
    supported = {str(name).replace("-", "_") for name in TARGET_DATASETS}
    if supported and slug not in supported:
        raise RuntimeError(
            f"this train-ci script supports {sorted(supported)}; got {dataset!r}. "
            "Generate a new PR with that dataset selected."
        )
    helpers = _load_helpers(slug)
    module_name = TRAINERS.get(slug) or TRAINERS.get(dataset) or REGRESSION.get(slug)
    candidates = [module_name, REGRESSION.get(slug), f"experiments.train_{slug}", f"train_{slug}", "experiments.train", "train"]
    train_mod = _import_first(candidates)
    if train_mod is not None:
        via_grid = _run_experiment_grid(train_mod, helpers, dataset=dataset, seed=seed, root=root)
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
        for name in HISTORY_NAMES:
            loaded = _metrics_from_history(history / name)
            if loaded is not None:
                return loaded
    raise RuntimeError(f"train-ci produced no metrics for dataset={dataset!r} seed={seed}")


if __name__ == "__main__":
    dataset, seed, output = _ci_env()
    acc, params = run_one(dataset=dataset, seed=seed, root=ROOT)
    extra = {
        "trainer": TRAINERS.get(dataset.replace("-", "_"), LAUNCH or ""),
        "helpers": HELPERS.get(dataset.replace("-", "_"), []),
        "schedule_id": PREFERRED_SCHEDULE_ID,
        "target_datasets": list(TARGET_DATASETS),
    }
    write_metrics(acc, params, extra=extra, dataset=dataset, seed=seed, output=output)
