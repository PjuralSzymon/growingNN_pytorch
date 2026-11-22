"""Build small stratified loaders used for fast simulation rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from growingnn.simulation.simulation_sets.base import dataset_labels
from growingnn.simulation.simulation_sets.stratified_random import StratifiedRandomSimulationSet

_CURRENT = StratifiedRandomSimulationSet()


def protected_sampling_indices(
    labels: torch.Tensor | np.ndarray,
    n: int,
    seed: int = 0,
) -> list[int]:
    """Class-balanced sample used by the current simulation set."""
    return _CURRENT.select_indices(labels, n, seed)


def _select_at_indices(x: Any, y: Any, indices: list[int]) -> tuple[Any, Any]:
    if torch.is_tensor(x):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return x[idx], y[idx]
    return np.asarray(x)[indices], np.asarray(y)[indices]


def create_simulation_set_sample(
    x: Any,
    y: Any,
    amount: int = 20,
    seed: int = 0,
) -> tuple[Any, Any]:
    indices = protected_sampling_indices(y, amount, seed)
    return _select_at_indices(x, y, indices)


create_simulation_set_SAMLE = create_simulation_set_sample


def sample_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    size: int,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build simulation loaders with the current stratified sampler."""
    return _CURRENT.sample_loaders(train_loader, val_loader, size, seed)


__all__ = [
    "create_simulation_set_SAMLE",
    "create_simulation_set_sample",
    "dataset_labels",
    "protected_sampling_indices",
    "sample_loaders",
]
