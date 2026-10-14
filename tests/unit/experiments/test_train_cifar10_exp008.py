"""Unit tests for the Experiment 008 generation-only recipe."""

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.train_cifar10_exp008_generation_only import (
    CHANNELS,
    EPOCHS,
    GENERATIONS,
    HIDDEN,
    SEEDS,
    CosineDecayScheduler,
    history_from_generations,
    parse_args,
    with_cutout,
)


def test_parse_args_defaults_to_the_three_experiment_seeds():
    """
    Experiment 008 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args([])

    # Assert
    assert args.seeds == SEEDS
    assert args.seeds == (110, 111, 112)


def test_parse_args_rejects_a_seed_outside_the_experiment():
    """
    A seed outside 110, 111, and 112 should be rejected.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--seeds", "1"])


def test_recipe_is_the_twelve_by_ten_cosine_generation_loop():
    """
    The fixed recipe should be 12 generations of 10 epochs on a 128-channel starter.
    """

    # Arrange
    scheduler = CosineDecayScheduler(GENERATIONS * EPOCHS)

    # Act
    learning_rate = scheduler.alpha_scheduler(0, 1)

    # Assert
    assert GENERATIONS == 12
    assert EPOCHS == 10
    assert CHANNELS == 128
    assert HIDDEN == 256
    assert scheduler.total_epochs == 120
    assert learning_rate == pytest.approx(0.1)


def test_history_marks_official_test_at_generation_ends_without_finish_chunks():
    """
    Saved history should place each official test at the end of its generation and omit finish chunks.
    """

    # Arrange
    summary = {
        "train_acc": [0.1, 0.2, 0.3, 0.4],
        "train_loss": [1.0, 0.9, 0.8, 0.7],
        "val_acc": [0.1, 0.2, 0.3, 0.4],
        "val_loss": [1.0, 0.9, 0.8, 0.7],
        "lr": [0.1, 0.1, 0.09, 0.09],
        "param_count": [10, 10, 20, 20],
        "test_acc": [0.15, 0.35],
        "test_loss": [1.1, 0.8],
        "action_generation": [0],
        "action_name": ["AddSeqConvLayer"],
        "action_val_acc_before": [0.2],
        "action_val_acc_after": [0.2],
    }

    # Act
    history = history_from_generations(summary, seed=110, epochs_per_generation=2)

    # Assert
    assert history["test_epoch"] == [2, 4]
    assert history["test_acc"] == [0.15, 0.35]
    assert "best_chunk" not in history
    assert "best_finish_test_acc" not in history


def test_with_cutout_keeps_the_training_set_length():
    """
    Cutout should wrap the training loader without dropping images.
    """

    # Arrange
    images = torch.zeros(6, 3, 32, 32)
    targets = torch.zeros(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=True)

    # Act
    wrapped = with_cutout(loader)

    # Assert
    assert len(wrapped.dataset) == 6
