"""Unit tests for the train-ci MNIST composed_step gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_CI_PATH = _REPO_ROOT / "ci" / "train_ci.py"


def _load_train_ci():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("train_ci", _TRAIN_CI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_fake_history(definition, hps) -> tuple[int, int]:
    hp = next(iter(hps))
    folder = definition.folder_name(hp)
    run_dir = definition.runs_dir / folder / f"seed_{definition.seeds[0]}"
    run_dir.mkdir(parents=True)
    torch.save(
        {"val_acc": [0.11, 0.87], "param_count": [100, 250]},
        run_dir / definition.history_filename,
    )
    return 1, 0


def test_mnist_hyperparameters_use_exp004_composed_step_cell():
    """
    mnist_hyperparameters should reuse Experiment 004 composed_step settings.
    """
    # Arrange
    train_ci = _load_train_ci()

    # Act
    hp = train_ci.mnist_hyperparameters()

    # Assert
    assert hp["epochs"] == train_ci.EPOCHS_PER_GENERATION
    assert hp["generations"] == train_ci.GENERATIONS
    assert hp["simulation_time"] == train_ci.SIMULATION_TIME_SEC
    assert hp["lr_alpha"] == train_ci.INITIAL_LR
    assert hp["score_accuracy_metric"] == train_ci.SCORE_ACCURACY_METRIC
    assert callable(hp["lr_scheduler_factory"])


def test_run_mnist_trains_one_seed_and_reads_history(tmp_path, monkeypatch):
    """
    run_mnist should run the experiment grid for the CI seed and return history metrics.
    """
    # Arrange
    train_ci = _load_train_ci()
    captured: dict[str, object] = {}

    def fake_grid(definition, hps, *, device, board):
        captured["seeds"] = tuple(definition.seeds)
        captured["model_factory"] = definition.model_factory
        captured["board"] = board
        captured["history_filename"] = definition.history_filename
        return _write_fake_history(definition, hps)

    monkeypatch.setattr(train_ci, "configure_deterministic_seeding", lambda: None)
    monkeypatch.setattr(train_ci.train_mnist, "MNISTData", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(train_ci.common, "run_experiment_grid", fake_grid)

    # Act
    acc, params = train_ci.run_mnist(seed=100, root=tmp_path)

    # Assert
    assert captured["seeds"] == (100,)
    assert captured["model_factory"] is train_ci.MODEL_FACTORY
    assert captured["board"] is False
    assert captured["history_filename"] == train_ci.train_mnist.MNIST_HISTORY_FILENAME
    assert acc == 0.87
    assert params == 250


def test_run_mnist_patches_always_scheduler_with_slope_gate(tmp_path, monkeypatch):
    """
    run_mnist should train under SlopeEstimationSimulationScheduler at 3 degrees.
    """
    # Arrange
    train_ci = _load_train_ci()
    captured: dict[str, object] = {}

    def fake_grid(definition, hps, *, device, board):
        captured["scheduler"] = train_ci.common.AlwaysSimulationScheduler
        return _write_fake_history(definition, hps)

    monkeypatch.setattr(train_ci, "configure_deterministic_seeding", lambda: None)
    monkeypatch.setattr(train_ci.train_mnist, "MNISTData", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(train_ci.common, "run_experiment_grid", fake_grid)

    # Act
    train_ci.run_mnist(seed=101, root=tmp_path)

    # Assert
    scheduler = captured["scheduler"]
    assert scheduler.func is train_ci.SlopeEstimationSimulationScheduler
    assert scheduler.keywords["angle_threshold"] == train_ci.SLOPE_ANGLE_THRESHOLD


def test_run_one_dispatches_mnist_to_composed_step_runner(tmp_path, monkeypatch):
    """
    run_one should send the mnist dataset through run_mnist.
    """
    # Arrange
    train_ci = _load_train_ci()
    monkeypatch.setattr(train_ci, "run_mnist", lambda **_kwargs: (0.5, 7))

    # Act
    acc, params = train_ci.run_one(dataset="mnist", seed=0, root=tmp_path)

    # Assert
    assert (acc, params) == (0.5, 7)


def test_run_mnist_raises_when_history_is_missing(tmp_path, monkeypatch):
    """
    run_mnist should fail instead of writing zero metrics when training produced no history.
    """
    # Arrange
    train_ci = _load_train_ci()
    monkeypatch.setattr(train_ci, "configure_deterministic_seeding", lambda: None)
    monkeypatch.setattr(train_ci.train_mnist, "MNISTData", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(
        train_ci.common, "run_experiment_grid", lambda *_a, **_k: (0, 0)
    )

    # Act / Assert
    try:
        train_ci.run_mnist(seed=0, root=tmp_path)
    except RuntimeError as exc:
        assert "did not write history" in str(exc)
    else:
        raise AssertionError("run_mnist should raise when history is missing")


def test_write_metrics_prints_hostinger_result_line(tmp_path, capsys):
    """
    write_metrics should persist JSON and print the Hostinger REGRESSION_CI_RESULT line.
    """
    # Arrange
    train_ci = _load_train_ci()
    output = tmp_path / "out.json"

    # Act
    train_ci.write_metrics(
        0.9,
        12,
        extra={"trainer": "experiments.train_mnist"},
        dataset="mnist",
        seed=0,
        output=output,
    )

    # Assert
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metric_value"] == 0.9
    assert payload["param_count"] == 12
    assert payload["seed"] == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(train_ci.RESULT_PREFIX)
    printed = json.loads(line[len(train_ci.RESULT_PREFIX) :])
    assert printed == {"dataset": "mnist", "seeds": [0], "val_acc": [0.9], "param_count": [12]}
