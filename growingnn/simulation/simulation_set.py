"""Build small stratified loaders used for fast simulation rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from growingnn.core.logger import logger

#TODO: Very old code, research for better way to create simualtion set 

def _dataset_labels(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets)
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return dataset.tensors[1]
    labels: list[Any] = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(item[1] if isinstance(item, tuple) else item[-1])
    return torch.as_tensor(labels)


def protected_sampling_indices(
    labels: torch.Tensor | np.ndarray,
    n: int,
    seed: int = 0,
) -> list[int]:
    labels = torch.as_tensor(labels)
    generator = torch.Generator().manual_seed(seed)
    unique_classes = torch.unique(labels)
    samples_per_class = max(1, n // len(unique_classes))
    selected: list[int] = []
    for cls in unique_classes.tolist():
        class_indices = torch.nonzero(labels == cls, as_tuple=False).view(-1)
        k = min(samples_per_class, len(class_indices))
        pick = class_indices[torch.randperm(len(class_indices), generator=generator)[:k]]
        selected.extend(pick.tolist())
    if len(selected) > n:
        logger.info("Simulation sample count %s exceeds requested %s (classes=%s, min one per class)", len(selected), n, len(unique_classes))
    return selected


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
    train_labels = _dataset_labels(train_loader.dataset)
    val_labels = _dataset_labels(val_loader.dataset)
    val_size = min(max(size // 4, 1), len(val_loader.dataset))
    train_idx = protected_sampling_indices(train_labels, min(size, len(train_loader.dataset)), seed)
    val_idx = protected_sampling_indices(val_labels, val_size, seed + 1)
    sim_train = DataLoader(
        Subset(train_loader.dataset, train_idx),
        batch_size=train_loader.batch_size,
        shuffle=True,
    )
    sim_val = DataLoader(
        Subset(val_loader.dataset, val_idx),
        batch_size=val_loader.batch_size,
    )
    return sim_train, sim_val
