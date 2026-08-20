"""Class-balanced random simulation set. Default GrowingNN sampler."""

from __future__ import annotations

from torch.utils.data import DataLoader

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.commons import (
    dataset_labels,
    indices_to_loaders,
    protected_sampling_indices,
)


class ProtectedSimulationSet(SimulationSet):
    """Class-balanced random subset. Same behavior as the old sample_loaders path."""

    def generate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int = 0,
        model=None,
    ) -> tuple[DataLoader, DataLoader]:
        train_labels = dataset_labels(train_loader.dataset)
        val_labels = dataset_labels(val_loader.dataset)
        val_size = min(max(size // 4, 1), len(val_loader.dataset))
        train_idx = protected_sampling_indices(train_labels, min(size, len(train_loader.dataset)), seed)
        val_idx = protected_sampling_indices(val_labels, val_size, seed + 1)
        return indices_to_loaders(train_loader, val_loader, train_idx, val_idx)
