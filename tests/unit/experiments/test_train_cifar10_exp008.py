"""Unit tests for the Experiment 008 generation-only recipe."""

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

from growingnn.core.config import RunningConfig
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from experiments.train_cifar10_exp008_generation_only import (
    CHANNELS,
    EPOCHS,
    GENERATIONS,
    HIDDEN,
    PHASE3_SIMULATION,
    SEEDS,
    CosineDecayScheduler,
    apply_action_group,
    apply_simulation_budget,
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
    assert args.phase == 1


def test_parse_args_rejects_a_seed_outside_the_experiment():
    """
    A seed outside 110, 111, and 112 should be rejected.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--seeds", "1"])


def test_parse_args_phase_two_defaults_to_the_three_action_groups():
    """
    Phase 2 should run the all, grow, and seq_res groups on the experiment seeds.
    """

    # Arrange / Act
    args = parse_args(["--phase", "2"])

    # Assert
    assert args.phase == 2
    assert args.groups == ("all", "grow", "seq_res")
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_three_defaults_to_the_experiment_seeds():
    """
    Phase 3 should train seeds 110, 111, and 112 unless a subset is requested.
    """

    # Arrange / Act
    args = parse_args(["--phase", "3"])

    # Assert
    assert args.phase == 3
    assert args.seeds == (110, 111, 112)


def test_parse_args_phase_three_rejects_an_action_group_list():
    """
    Phase 3 should reject --groups because it always uses the all-action list.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        parse_args(["--phase", "3", "--groups", "grow"])


def test_phase_three_budget_is_one_hundred_seconds_on_the_all_group():
    """
    The phase 3 budget should be 100 seconds, 8 epochs, and 2000 images, with every action still enabled.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "all")

    # Act
    apply_simulation_budget(cfg, PHASE3_SIMULATION)

    # Assert
    assert PHASE3_SIMULATION == (100.0, 8, 2000)
    assert isinstance(cfg.simulation_scheduler, AlwaysSimulationScheduler)
    assert cfg.simulation_scheduler.simulation_time == 100.0
    assert cfg.simulation_scheduler.simulation_epochs == 8
    assert cfg.simulation_set_size == 2000
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True


def test_grow_group_keeps_add_neurons_and_turns_delete_off():
    """
    The grow group should allow layer adds and neuron adds, and disallow deletes.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)

    # Act
    apply_action_group(cfg, "grow")

    # Assert
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_11 is True
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is False
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01 is False


def test_seq_res_group_turns_add_neurons_off():
    """
    The seq_res group should allow sequential and residual adds, and disallow neuron adds.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)

    # Act
    apply_action_group(cfg, "seq_res")

    # Assert
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_11 is False
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_05 is False


def test_all_group_enables_delete_and_dropout():
    """
    The all group should leave every structure-action flag on.
    """

    # Arrange
    cfg = RunningConfig(generations=1, epochs=1)
    apply_action_group(cfg, "seq_conv")

    # Act
    apply_action_group(cfg, "all")

    # Assert
    assert cfg.ACTIONS_ENABLE_DEL_LAYER is True
    assert cfg.ACTIONS_ENABLE_DEL_NEURONS_09 is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05 is True
    assert cfg.ACTIONS_ENABLE_ADD_NEURONS_20 is True
    assert cfg.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True


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
