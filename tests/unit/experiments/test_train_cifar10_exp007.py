"""Unit tests for the Experiment 007 original versus added-package settings."""

from __future__ import annotations

import torch.nn as nn

import pytest
import torch

from experiments.train_cifar10_exp007_test90_package import (
    ADDED_FINISH_CHUNKS,
    ADDED_FINISH_DECAY_AT_001,
    ADDED_FINISH_DECAY_AT_01,
    ADDED_FINISH_EPOCHS_PER_CHUNK,
    ADDED_LABEL_SMOOTHING,
    SEEDS,
    FourLayerCifarNet,
    added_history_from_growth,
    append_finish_chunk,
    build_model,
    finish_added_training,
    parse_args,
    running_config_for_group,
)


def test_original_group_keeps_the_plain_recipe():
    """
    The original group should use the plain starter and leave the new conv package off.
    """

    # Arrange / Act
    model = build_model("original", {"model_channels": 32, "model_hidden_dim": 64})
    cfg = running_config_for_group("original")

    # Assert
    assert isinstance(model, nn.Sequential) is False
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert cfg.seq_conv_insert_before_global_pool is False
    assert cfg.max_grad_norm is None
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.generations == 8
    assert cfg.epochs == 10


def test_added_group_enables_the_package_that_reached_90():
    """
    The added group should use the BatchNorm starter, before-pool convs, clip, and seq-conv only.
    """

    # Arrange / Act
    model = build_model("added", {"model_channels": 64, "model_hidden_dim": 128})
    cfg = running_config_for_group("added")

    # Assert
    assert isinstance(model, FourLayerCifarNet)
    assert any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert cfg.seq_conv_insert_before_global_pool is True
    assert cfg.max_grad_norm == 1.0
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is False
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_LAYER is False
    assert cfg.ACTIONS_ENABLE_ADD_RES_LAYER is False
    assert cfg.generations == 7
    assert cfg.epochs == 8
    assert ADDED_FINISH_DECAY_AT_01 == 30
    assert ADDED_FINISH_DECAY_AT_001 == 60
    assert ADDED_LABEL_SMOOTHING == 0.1


def test_experiment_trains_three_seeds():
    """
    Experiment 007 should use seeds 110, 111, and 112 for both groups.
    """

    # Arrange / Act / Assert
    assert SEEDS == (110, 111, 112)


def test_parse_args_selects_only_the_added_group():
    """
    --group added should rerun the added package and keep the three experiment seeds.
    """

    # Arrange / Act
    args = parse_args(["--group", "added"])

    # Assert
    assert args.group == "added"
    assert args.seeds == SEEDS


def test_parse_args_accepts_one_experiment_seed():
    """
    --seeds should limit the rerun to a seed from the experiment list.
    """

    # Arrange / Act
    args = parse_args(["--group", "added", "--seeds", "111"])

    # Assert
    assert args.seeds == (111,)


def test_parse_args_rejects_a_seed_outside_the_experiment():
    """
    parse_args should reject a seed that is not 110, 111, or 112.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--seeds", "1"])


def test_added_history_from_growth_marks_generation_end_tests():
    """
    Growth official tests should be stored at the last epoch of each generation.
    """

    # Arrange
    summary = {
        "train_acc": [0.1, 0.2, 0.3, 0.4],
        "val_acc": [0.15, 0.25, 0.35, 0.45],
        "train_loss": [1.0, 0.9, 0.8, 0.7],
        "val_loss": [1.1, 1.0, 0.9, 0.8],
        "lr": [0.1, 0.1, 0.1, 0.1],
        "param_count": [10, 10, 20, 20],
        "test_acc": [0.3, 0.4],
        "test_loss": [0.9, 0.8],
        "action_generation": [0],
        "action_name": ["AddSeqConvLayer"],
        "action_val_acc_before": [0.2],
        "action_val_acc_after": [0.2],
    }

    # Act
    history = added_history_from_growth(summary, seed=110, epochs_per_generation=2)

    # Assert
    assert history["train_acc"] == [0.1, 0.2, 0.3, 0.4]
    assert history["val_acc"] == [0.15, 0.25, 0.35, 0.45]
    assert history["test_acc"] == [0.3, 0.4]
    assert history["test_epoch"] == [2, 4]
    assert history["action_name"] == ["AddSeqConvLayer"]


def test_append_finish_chunk_keeps_epoch_train_val_and_chunk_test():
    """
    A finish chunk should add its train and validation epochs and one official test at the chunk end.
    """

    # Arrange
    history = added_history_from_growth(
        {
            "train_acc": [0.2] * 8,
            "val_acc": [0.1] * 8,
            "train_loss": [1.0] * 8,
            "val_loss": [1.1] * 8,
            "lr": [0.1] * 8,
            "param_count": [48] * 8,
            "test_acc": [0.5],
            "test_loss": [0.8],
        },
        seed=110,
        epochs_per_generation=8,
    )
    chunk_history = {
        "train_acc": [0.6, 0.61, 0.62, 0.63, 0.64],
        "val_acc": [0.55, 0.56, 0.57, 0.58, 0.59],
        "train_loss": [0.5] * 5,
        "val_loss": [0.6] * 5,
        "lr": [0.1] * 5,
    }

    # Act
    append_finish_chunk(history, chunk_history, chunk=0, test_acc=0.7, test_loss=0.4)

    # Assert
    assert len(history["train_acc"]) == 13
    assert history["train_acc"][-1] == 0.64
    assert history["val_acc"][-1] == 0.59
    assert history["test_acc"] == [0.5, 0.7]
    assert history["test_epoch"] == [8, 13]
    assert history["best_chunk"] == 0
    assert history["best_epoch"] == 13


def test_finish_added_training_writes_train_val_and_test(monkeypatch, tmp_path):
    """
    history.pt should keep growth and every finish epoch of train, validation, and official test.
    """

    # Arrange
    import growingnn.training.gradient_descent as gradient_descent

    def fake_gradient_descent(model, epochs, *args, **kwargs):
        assert epochs == ADDED_FINISH_EPOCHS_PER_CHUNK
        return model, {
            "train_acc": [0.8] * epochs,
            "val_acc": [0.7] * epochs,
            "train_loss": [0.4] * epochs,
            "val_loss": [0.5] * epochs,
            "lr": [0.1] * epochs,
        }

    monkeypatch.setattr(gradient_descent, "gradient_descent", fake_gradient_descent)
    monkeypatch.setattr(gradient_descent, "_evaluate", lambda *args, **kwargs: (0.2, 0.91))
    history = {
        "seed": 110,
        "growth_epochs": 8,
        "train_loss": [1.0] * 8,
        "train_acc": [0.4] * 8,
        "val_loss": [1.1] * 8,
        "val_acc": [0.3] * 8,
        "lr": [0.1] * 8,
        "test_acc": [0.5],
        "test_loss": [0.9],
        "test_epoch": [8],
        "best_finish_test_acc": 0.0,
        "best_chunk": None,
        "best_epoch": None,
    }

    # Act
    finish_added_training(nn.Linear(2, 2), None, None, None, tmp_path, seed=110, history=history)
    saved = torch.load(tmp_path / "history.pt", weights_only=False)

    # Assert
    assert len(saved["train_acc"]) == 8 + ADDED_FINISH_CHUNKS * ADDED_FINISH_EPOCHS_PER_CHUNK
    assert len(saved["val_acc"]) == len(saved["train_acc"])
    assert saved["test_epoch"][0] == 8
    assert saved["test_epoch"][-1] == 8 + ADDED_FINISH_CHUNKS * ADDED_FINISH_EPOCHS_PER_CHUNK
    assert len(saved["test_acc"]) == 1 + ADDED_FINISH_CHUNKS
    assert saved["best_epoch"] == 13


def test_finish_added_training_continues_after_test_passes_90(monkeypatch, tmp_path):
    """
    finish_added_training should run every finish chunk after official test accuracy passes 0.90.
    """

    # Arrange
    import growingnn.training.gradient_descent as gradient_descent

    calls = {"n": 0}

    def fake_gradient_descent(model, epochs, *args, **kwargs):
        calls["n"] += 1
        assert epochs == ADDED_FINISH_EPOCHS_PER_CHUNK
        return model, {"train_acc": [0.91], "val_acc": [0.9]}

    monkeypatch.setattr(gradient_descent, "gradient_descent", fake_gradient_descent)
    monkeypatch.setattr(gradient_descent, "_evaluate", lambda *args, **kwargs: (0.2, 0.95))
    model = nn.Linear(2, 2)

    # Act
    best = finish_added_training(model, None, None, None, tmp_path, seed=110)

    # Assert
    assert calls["n"] == ADDED_FINISH_CHUNKS
    assert ADDED_FINISH_CHUNKS == 24
    assert best == 0.95
