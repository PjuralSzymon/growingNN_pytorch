"""Checks for the Experiment 013 MCTS settings and simulation-set size."""

import growingnn.core.config as project_config
from experiments.train_cifar10_exp013_residual_block_action_grading import (
    MCTS_SETTINGS,
    TARGET_VALIDATION_ACCURACY,
    apply_experiment_010_mcts_settings,
    restore_mcts_settings,
    simulation_sample_size,
)
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler


def test_validation_stop_is_ninety_percent():
    """
    Experiment 013 should stop training when validation accuracy reaches 90%.
    """

    # Arrange
    expected_accuracy = 0.90

    # Act
    target_accuracy = TARGET_VALIDATION_ACCURACY

    # Assert
    assert target_accuracy == expected_accuracy


def test_simulation_sample_size_uses_ten_percent_of_each_clean_training_split():
    """
    simulation_sample_size should use 10% of the MNIST and CIFAR-10 clean training splits.
    """

    # Arrange
    mnist_clean_train = 50_000
    cifar10_clean_train = 45_000

    # Act
    mnist_size = simulation_sample_size(mnist_clean_train)
    cifar10_size = simulation_sample_size(cifar10_clean_train)

    # Assert
    assert mnist_size == 5_000
    assert cifar10_size == 4_500


def test_apply_experiment_010_mcts_settings_installs_pass2_winners():
    """
    apply_experiment_010_mcts_settings should install the Experiment 010 pass-2 MCTS winners.
    """

    # Arrange
    previous = {name: getattr(project_config, name) for name in MCTS_SETTINGS}

    # Act
    try:
        apply_experiment_010_mcts_settings()
        installed_c = project_config.MCTS_UCB1_C
        installed_depth = project_config.MCTS_ROLLOUT_DEPTH
        installed_propagate = project_config.MCTS_PROPAGATE_ROLLOUT_VALUE
        installed_lr = project_config.MCTS_ROLLOUT_LR
    finally:
        restore_mcts_settings(previous)

    # Assert
    assert installed_c == 1.0
    assert installed_depth == 1
    assert installed_propagate is True
    assert isinstance(installed_lr, ActionLearningRateScheduler)
    assert installed_lr.learning_rate_config_board_labels()[1] == 0.0001


def test_restore_mcts_settings_puts_back_the_previous_globals():
    """
    restore_mcts_settings should return every changed MCTS global to its previous value.
    """

    # Arrange
    previous = apply_experiment_010_mcts_settings()

    # Act
    restore_mcts_settings(previous)

    # Assert
    for name, value in previous.items():
        assert getattr(project_config, name) is value
