"""Unit tests for the shared experiment execution lifecycle."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import experiments_common as common
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    NeverSimulationScheduler,
)


def _hyperparameters() -> dict[str, object]:
    return {
        "generations": 1,
        "epochs": 1,
        "batch_size": 2,
        "lr_alpha": 0.01,
        "simulation_time": 1.0,
        "simulation_epochs": 1,
        "simulation_set_size": 2,
        "target_accuracy": 0.99,
        "score_weight_acc": 1.0,
        "score_weight_countw": 0.1,
    }


def _loaders(_hp: dict[str, object]) -> common.LoaderSet:
    dataset = TensorDataset(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    return loader, loader, loader


def _definition(tmp_path, **overrides) -> common.ExperimentDefinition:
    values = {
        "name": "test",
        "runs_dir": tmp_path / "runs",
        "history_filename": "history.pt",
        "seeds": [7],
        "folder_name": lambda _hp: "config",
        "model_factory": lambda _hp: nn.Linear(2, 2),
        "loader_factory": _loaders,
        "board_metadata": lambda _hp, folder, seed: (f"{folder}-{seed}", "test"),
    }
    values.update(overrides)
    return common.ExperimentDefinition(**values)


def test_parse_board_cli_converts_false_to_boolean(monkeypatch):
    """
    parse_board_cli should expose --board false as a boolean False value.
    """
    # Arrange
    monkeypatch.setattr(sys, "argv", ["experiment.py", "--board", "false"])

    # Act
    args = common.parse_board_cli("test experiment")

    # Assert
    assert args.board is False


def test_require_cuda_rejects_unavailable_cuda(monkeypatch):
    """
    require_cuda should stop a CUDA-only experiment when CUDA is unavailable.
    """
    # Arrange
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Act / Assert
    try:
        common.require_cuda(torch.device("cuda"))
    except RuntimeError as exc:
        assert "requires CUDA" in str(exc)
    else:
        raise AssertionError("require_cuda should raise RuntimeError")


def test_run_experiment_grid_skips_existing_seed_folder(tmp_path):
    """
    run_experiment_grid should not build or train a run whose seed folder exists.
    """
    # Arrange
    model_factory = MagicMock(return_value=nn.Linear(2, 2))
    definition = _definition(tmp_path, model_factory=model_factory)
    (definition.runs_dir / "config" / "seed_7").mkdir(parents=True)

    # Act
    result = common.run_experiment_grid(
        definition,
        [_hyperparameters()],
        device=torch.device("cpu"),
        board=False,
    )

    # Assert
    assert result == (0, 1)
    model_factory.assert_not_called()


def test_run_experiment_grid_trains_and_saves_metric_artifacts(tmp_path, monkeypatch):
    """
    run_experiment_grid should train a missing run and save its history and metric plots.
    """
    # Arrange
    summary = {
        "generation": [0],
        "train_loss": [1.0],
        "train_acc": [0.5],
        "val_loss": [0.8],
        "val_acc": [0.6],
        "lr": [0.01],
        "param_count": [6],
    }
    captured = {}

    def _fake_train(model, _train, _val, config, **_kwargs):
        captured["config"] = config
        return model, summary

    monkeypatch.setattr(common, "train_generations", _fake_train)
    monkeypatch.setattr(
        common,
        "sample_loaders",
        lambda clean_train, val, _size, *, seed: (clean_train, val),
    )
    definition = _definition(tmp_path)

    # Act
    result = common.run_experiment_grid(
        definition,
        [_hyperparameters()],
        device=torch.device("cpu"),
        board=False,
    )

    # Assert
    run_dir = definition.runs_dir / "config" / "seed_7"
    assert result == (1, 0)
    assert torch.load(run_dir / "history.pt", weights_only=True) == {
        key: summary[key] for key in common.METRIC_KEYS
    }
    assert {path.name for path in run_dir.glob("*.png")} == {
        f"{key}.png" for key in common.METRIC_KEYS
    }
    assert captured["config"].generations == 1
    assert captured["config"].ACTIONS_ENABLE_ADD_NEURONS_15 is True
    assert captured["config"].ACTIONS_ENABLE_DEL_NEURONS_05 is True


def test_running_config_enables_neuron_resize_by_default():
    """
    Shared experiment runner keeps RunningConfig neuron-resize defaults on.
    Historical exp001-005 turn them off locally; CI and new experiments keep them on.
    """
    # Arrange
    hp = _hyperparameters()
    device = torch.device("cpu")

    # Act
    cfg = common._running_config(hp, device, None)

    # Assert
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_11 is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_15 is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_20 is True
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_01 is True
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_05 is True
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_09 is True


def test_running_config_defaults_momentum_weight_decay_and_always_scheduler():
    """
    _running_config should keep MNIST defaults: momentum 0, weight decay 0, and always-on search.
    """
    # Arrange
    hp = _hyperparameters()

    # Act
    cfg = common._running_config(hp, torch.device("cpu"), None)

    # Assert
    assert cfg.momentum == 0.0
    assert cfg.weight_decay == 0.0
    assert isinstance(cfg.simulation_scheduler, AlwaysSimulationScheduler)


def test_running_config_reads_optimizer_settings_and_scheduler_factory():
    """
    _running_config should copy momentum, weight decay, and a caller-built simulation scheduler.
    """
    # Arrange
    hp = _hyperparameters()
    hp["momentum"] = 0.9
    hp["weight_decay"] = 5e-4
    sentinel = NeverSimulationScheduler()
    hp["simulation_scheduler_factory"] = lambda _hp: sentinel

    # Act
    cfg = common._running_config(hp, torch.device("cpu"), None)

    # Assert
    assert cfg.momentum == 0.9
    assert cfg.weight_decay == 5e-4
    assert cfg.simulation_scheduler is sentinel


def test_running_config_copies_grad_clip_and_conv_placement():
    """
    _running_config should copy a grad-norm cap and the sequential-conv placement switch.
    """
    # Arrange
    hp = _hyperparameters()
    hp["max_grad_norm"] = 1.0
    hp["seq_conv_insert_before_global_pool"] = False

    # Act
    cfg = common._running_config(hp, torch.device("cpu"), None)

    # Assert
    assert cfg.max_grad_norm == 1.0
    assert cfg.seq_conv_insert_before_global_pool is False


def test_running_config_can_disable_neuron_deletion():
    """
    _running_config should turn off neuron deletion when the hyperparameter flag is set.
    """
    # Arrange
    hp = _hyperparameters()
    hp["disable_del_neurons"] = True

    # Act
    cfg = common._running_config(hp, torch.device("cpu"), None)

    # Assert
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_01 is False
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_05 is False
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_09 is False
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True


def test_running_config_plain_growth_keeps_conv_and_linear_adds():
    """
    plain_growth should turn off shrink, neuron resize, and dropout, and leave conv and linear adds on.
    """
    # Arrange
    hp = _hyperparameters()
    hp["plain_growth"] = True

    # Act
    cfg = common._running_config(hp, torch.device("cpu"), None)

    # Assert
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is False
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_01 is False
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_11 is False
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01 is False
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_LAYER is True


def test_draw_graphs_uses_full_and_simplified_names(tmp_path, monkeypatch):
    """
    _draw_graphs should write full and simplified graph files with the same suffix.
    """
    # Arrange
    simplified = MagicMock()
    full = MagicMock()
    monkeypatch.setattr(common, "draw_filtered_fx_graph", simplified)
    monkeypatch.setattr(common, "draw_torch_fx_graph", full)
    gm = torch.fx.symbolic_trace(nn.Linear(2, 2))

    # Act
    common._draw_graphs(tmp_path, "_error", gm)

    # Assert
    simplified.assert_called_once_with(
        gm, str(tmp_path / "fx_graph_error_simplified"), fmt="pdf"
    )
    full.assert_called_once_with(gm, str(tmp_path / "fx_graph_error"), fmt="pdf")
