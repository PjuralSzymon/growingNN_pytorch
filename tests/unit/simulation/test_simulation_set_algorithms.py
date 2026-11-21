"""Unit tests for simulation-set selector classes."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.simulation.simulation_set import protected_sampling_indices
from growingnn.simulation.simulation_sets import (
    SIMULATION_SET_ALGORITHMS,
    CcsSimulationSet,
    El2nSimulationSet,
    ModerateSimulationSet,
    SimulationSetAlgorithm,
    StratifiedRandomSimulationSet,
)


def test_every_simulation_set_algorithm_subclasses_the_base():
    """
    Each selector class should inherit the shared simulation-set contract.
    """

    # Arrange
    algorithms = SIMULATION_SET_ALGORITHMS

    # Act
    inheritance = [issubclass(algorithm, SimulationSetAlgorithm) for algorithm in algorithms]

    # Assert
    assert len(algorithms) == 4
    assert all(inheritance)


def test_stratified_random_matches_the_current_sampler():
    """
    StratifiedRandomSimulationSet should reproduce protected_sampling_indices.
    """

    # Arrange
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    selector = StratifiedRandomSimulationSet()

    # Act
    selected = selector.select_indices(labels, size=6, seed=0)

    # Assert
    assert selected == protected_sampling_indices(labels, n=6, seed=0)


def test_el2n_keeps_the_highest_score_in_each_class():
    """
    EL2N selection should keep the largest score inside each class.
    """

    # Arrange
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    scores = torch.tensor([0.1, 5.0, 0.2, 0.3, 4.0, 0.4])
    selector = El2nSimulationSet()

    # Act
    selected = selector.select_indices(labels, size=2, seed=0, scores=scores)

    # Assert
    assert selected == [1, 4]


def test_ccs_drops_the_hardest_scores_and_keeps_an_easy_example():
    """
    CCS should exclude the hardest tail and still keep a low-score example.
    """

    # Arrange
    labels = torch.zeros(10)
    scores = torch.arange(10).float()
    selector = CcsSimulationSet(strata=2, hard_cutoff=0.3)

    # Act
    selected = selector.select_indices(labels, size=4, seed=0, scores=scores)

    # Assert
    assert len(selected) == 4
    assert set(selected).isdisjoint({7, 8, 9})
    assert min(selected) <= 2


def test_moderate_keeps_the_example_nearest_the_median_distance():
    """
    Moderate selection should keep the example closest to the median class-center distance.
    """

    # Arrange
    labels = torch.zeros(5)
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0]])
    selector = ModerateSimulationSet()

    # Act
    selected = selector.select_indices(labels, size=1, seed=0, features=features)

    # Assert
    assert selected == [1]


def test_el2n_sample_loaders_cover_every_class():
    """
    EL2N sample_loaders should return a train loader that still contains every class.
    """

    # Arrange
    features = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.2]])
    labels = torch.tensor([0, 0, 1, 1])
    train = DataLoader(TensorDataset(features, labels), batch_size=2)
    val = DataLoader(TensorDataset(features, labels), batch_size=2)
    selector = El2nSimulationSet(epochs=1)

    # Act
    sim_train, _sim_val = selector.sample_loaders(train, val, size=2, seed=0)
    train_labels = torch.cat([batch[1] for batch in sim_train])

    # Assert
    assert len(torch.unique(train_labels)) == 2
