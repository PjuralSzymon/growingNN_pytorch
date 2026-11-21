"""Unit tests for the Experiment 011 simulation-set comparison."""

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import growingnn.core.config as project_config
from experiments.train_cifar10_exp011_simulation_sets import (
    ALGORITHM_NAMES,
    GENERATIONS,
    EPOCHS,
    PASS2_MCTS,
    SIMULATION_TIME_SECONDS,
    apply_mcts_settings,
    build_selector,
    initial_graph,
    parse_args,
    requested_simulation_size,
    run_directory,
    run_one,
    selected_datasets,
    selected_seeds,
    summarize_results,
)
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_sets import (
    CcsSimulationSet,
    El2nSimulationSet,
    ModerateSimulationSet,
    StratifiedRandomSimulationSet,
)


def test_algorithm_list_keeps_the_old_sampler_and_the_three_remaining_rules():
    """
    The comparison grid should contain stratified random, EL2N, CCS, and moderate.
    """

    # Arrange / Act
    names = ALGORITHM_NAMES

    # Assert
    assert names == (
        "stratified_random",
        "el2n",
        "ccs",
        "moderate",
    )


def test_requested_simulation_size_is_ten_percent_of_the_clean_split():
    """
    A 45,000-image clean split at 10% should request 4,500 simulation training images.
    """

    # Arrange / Act
    size = requested_simulation_size(45_000)

    # Assert
    assert size == 4_500


def test_build_selector_returns_the_named_class():
    """
    build_selector should construct the class registered under that algorithm name.
    """

    # Arrange
    expected = {
        "stratified_random": StratifiedRandomSimulationSet,
        "el2n": El2nSimulationSet,
        "ccs": CcsSimulationSet,
        "moderate": ModerateSimulationSet,
    }

    # Act
    selectors = {name: build_selector(name) for name in expected}

    # Assert
    assert {name: type(selector) for name, selector in selectors.items()} == expected


def test_build_selector_rejects_an_unknown_algorithm():
    """
    build_selector should raise ValueError when the algorithm name is outside the grid.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unknown simulation-set algorithm"):
        build_selector("random_uniform")


def test_apply_mcts_settings_installs_the_pass2_winners():
    """
    apply_mcts_settings should put the Experiment 010 Pass 2 MCTS constants on the process config.
    """

    # Arrange
    names = tuple(PASS2_MCTS)
    original = {name: getattr(project_config, name) for name in names}

    # Act
    try:
        apply_mcts_settings()
        installed_c = project_config.MCTS_UCB1_C
        installed_depth = project_config.MCTS_ROLLOUT_DEPTH
        installed_propagation = project_config.MCTS_PROPAGATE_ROLLOUT_VALUE
        installed_minimum_runs = project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS
        installed_lr = project_config.MCTS_ROLLOUT_LR.learning_rate_config_board_labels()[1]
    finally:
        for name, value in original.items():
            setattr(project_config, name, value)

    # Assert
    assert installed_c == 1.0
    assert installed_depth == 1
    assert installed_propagation is True
    assert installed_minimum_runs == 3
    assert installed_lr == 0.0001


def test_selected_seeds_rejects_a_seed_outside_the_comparison():
    """
    Experiment 011 should stay on seeds 1 and 2 so each selector has the same seed pair.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="seeds must be in"):
        selected_seeds([1, 3])


def test_parse_args_defaults_to_every_algorithm_dataset_and_both_seeds():
    """
    An empty command line should select all four selectors, all three datasets, and both seeds.
    """

    # Arrange / Act
    args = parse_args([])

    # Assert
    assert args.algorithms is None
    assert args.datasets is None
    assert args.seeds is None


def test_selected_datasets_rejects_a_dataset_outside_the_grid():
    """
    selected_datasets should accept only CIFAR-10, MNIST, and CIFAR-100.
    """

    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="datasets must be in"):
        selected_datasets(["tiny_imagenet"])


def test_initial_graph_uses_the_dataset_channel_count_and_class_count():
    """
    MNIST should start from one input channel, and CIFAR-100 should end in 100 classes.
    """

    # Arrange / Act
    mnist = initial_graph("mnist")
    cifar100 = initial_graph("cifar100")

    # Assert
    assert mnist.conv1.in_channels == 1
    assert mnist.linear2.out_features == 10
    assert cifar100.conv1.in_channels == 3
    assert cifar100.linear2.out_features == 100


def test_run_directory_keeps_finished_cifar10_runs_at_the_original_path():
    """
    CIFAR-10 should keep the finished algorithm/seed path, and new datasets should add a dataset folder.
    """

    # Arrange / Act
    cifar10 = run_directory("cifar10", "moderate", 1)
    mnist = run_directory("mnist", "moderate", 1)

    # Assert
    assert cifar10.parts[-2:] == ("moderate", "seed_1")
    assert mnist.parts[-3:] == ("mnist", "moderate", "seed_1")


def test_summarize_results_ranks_the_higher_mean_validation_first():
    """
    summarize_results should order algorithms by mean final validation accuracy, highest first.
    """

    # Arrange
    runs = [
        {"dataset": "cifar10", "algorithm": "el2n", "final_validation_accuracy": 0.40},
        {"dataset": "cifar10", "algorithm": "el2n", "final_validation_accuracy": 0.50},
        {"dataset": "cifar10", "algorithm": "stratified_random", "final_validation_accuracy": 0.60},
        {"dataset": "cifar10", "algorithm": "stratified_random", "final_validation_accuracy": 0.70},
    ]

    # Act
    ranking = summarize_results(runs)

    # Assert
    assert [row["algorithm"] for row in ranking] == ["stratified_random", "el2n"]
    assert ranking[0]["mean_final_validation_accuracy"] == pytest.approx(0.65)


def test_run_one_skips_a_finished_folder(tmp_path):
    """
    run_one should return the saved result and skip training when history and run.json already exist.
    """

    # Arrange
    output_dir = tmp_path / "el2n" / "seed_1"
    output_dir.mkdir(parents=True)
    saved = {"algorithm": "el2n", "seed": 1, "final_validation_accuracy": 0.42}
    (output_dir / "history.pt").write_bytes(b"saved")
    (output_dir / "run.json").write_text(json.dumps(saved), encoding="utf-8")

    # Act
    result = run_one("el2n", 1, object(), output_dir)

    # Assert
    assert result["final_validation_accuracy"] == 0.42


def test_run_one_trains_with_the_selector_loaders_and_pass2_budget(tmp_path, monkeypatch):
    """
    run_one should score MCTS on the selector's loaders for 10 generations with a 32 second budget.
    """

    # Arrange
    features = torch.randn(20, 3, 8, 8)
    labels = torch.tensor([index % 2 for index in range(20)])
    clean = DataLoader(TensorDataset(features, labels), batch_size=4)
    val = DataLoader(TensorDataset(features[:8], labels[:8]), batch_size=4)
    test = DataLoader(TensorDataset(features[:4], labels[:4]), batch_size=2)
    captured = {}

    class _Data:
        def loaders(self, _batch_size):
            return clean, val, clean

        def test_loader(self, _batch_size):
            return test

    def _train(*args, **kwargs):
        captured["config"] = args[3]
        captured["sim_train"] = kwargs["sim_train_loader"]
        captured["sim_val"] = kwargs["sim_val_loader"]
        return args[0], {
            "train_acc": [0.2],
            "val_acc": [0.3],
            "test_acc": [0.25],
            "train_loss": [1.0],
            "val_loss": [1.1],
            "test_loss": [1.2],
            "lr": [0.1],
            "param_count": [296],
            "generation": [0],
        }

    monkeypatch.setattr(
        "experiments.train_cifar10_exp011_simulation_sets.train_generations",
        _train,
    )
    monkeypatch.setattr(
        "experiments.train_cifar10_exp011_simulation_sets.with_cutout",
        lambda loader: loader,
    )
    original = {name: getattr(project_config, name) for name in PASS2_MCTS}

    # Act
    try:
        result = run_one("stratified_random", 1, _Data(), tmp_path / "stratified_random" / "seed_1")
        config = captured["config"]
    finally:
        for name, value in original.items():
            setattr(project_config, name, value)

    # Assert
    assert result["final_validation_accuracy"] == pytest.approx(0.3)
    assert config.generations == GENERATIONS
    assert config.epochs == EPOCHS
    assert isinstance(config.simulation_scheduler, AlwaysSimulationScheduler)
    assert config.simulation_scheduler.simulation_time == SIMULATION_TIME_SECONDS
    assert config.simulation_scheduler.simulation_epochs == 1
    assert isinstance(config.criterion, nn.CrossEntropyLoss)
    assert len(captured["sim_train"].dataset) == requested_simulation_size(20)
    assert len(captured["sim_val"].dataset) >= 1
