"""Shared contract for simulation-set selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from growingnn.core.config import DATALOADER_NUM_WORKERS
from growingnn.core.logger import logger


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    """Read one label per example from a dataset or subset."""
    if isinstance(dataset, Subset):
        base_labels = dataset_labels(dataset.dataset)
        return base_labels[torch.as_tensor(list(dataset.indices), dtype=torch.long)]
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets)
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return dataset.tensors[1]
    labels = []
    for index in range(len(dataset)):
        item = dataset[index]
        labels.append(item[1] if isinstance(item, tuple) else item[-1])
    return torch.as_tensor(labels)


def dataset_matrix(dataset: Dataset) -> torch.Tensor:
    """Flatten each example into one row."""
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "tensors"):
        base = dataset.dataset.tensors[0]
        rows = base[torch.as_tensor(list(dataset.indices), dtype=torch.long)]
        return rows.reshape(rows.shape[0], -1).float()
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 1:
        rows = dataset.tensors[0]
        return rows.reshape(rows.shape[0], -1).float()
    rows = [
        torch.as_tensor(dataset[index][0]).reshape(-1).float()
        for index in range(len(dataset))
    ]
    if not rows:
        return torch.zeros(0, 1)
    return torch.stack(rows)


def class_center_distance(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the class center, distance of each row to it, and the median distance."""
    center = features.mean(dim=0)
    distances = (features - center).norm(dim=1)
    return center, distances, distances.median()


class SimulationSetAlgorithm(ABC):
    """Select a class-balanced subset and wrap it as simulation loaders."""

    name: str

    def bind_train(
        self,
        dataset: Dataset,
        seed: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return features and scores for the training split. The default uses labels only."""
        return None, None

    def bind_val(
        self,
        dataset: Dataset,
        seed: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return features and scores for the validation split."""
        return self.bind_train(dataset, seed)

    def select_indices(
        self,
        labels: torch.Tensor,
        size: int,
        seed: int = 0,
        *,
        features: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
    ) -> list[int]:
        """Pick up to ``size`` indices, split evenly across classes."""
        labels = torch.as_tensor(labels).reshape(-1)
        if features is not None:
            features = features.detach().float()
            if features.ndim == 1:
                features = features.unsqueeze(1)
            else:
                features = features.reshape(features.shape[0], -1)
        if scores is not None:
            scores = torch.as_tensor(scores).reshape(-1).float()
        unique_classes = torch.unique(labels)
        if len(unique_classes) == 0 or size <= 0:
            return []
        per_class = max(1, int(size) // len(unique_classes))
        selected: list[int] = []
        for offset, cls in enumerate(unique_classes.tolist()):
            class_indices = torch.nonzero(labels == cls, as_tuple=False).view(-1)
            quota = min(per_class, int(class_indices.numel()))
            class_features = None if features is None else features[class_indices]
            class_scores = None if scores is None else scores[class_indices]
            picked = self._select_class(
                class_indices,
                quota,
                seed + offset,
                class_features,
                class_scores,
            )
            selected.extend(int(index) for index in picked)
        if len(selected) > size:
            logger.info(
                "Simulation sample count %s exceeds requested %s (classes=%s, min one per class)",
                len(selected),
                size,
                len(unique_classes),
            )
        return selected

    @abstractmethod
    def _select_class(
        self,
        indices: torch.Tensor,
        quota: int,
        seed: int,
        features: torch.Tensor | None,
        scores: torch.Tensor | None,
    ) -> list[int]:
        """Return ``quota`` dataset indices belonging to one class."""

    def sample_loaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        """Build the simulation train and validation loaders for this selector."""
        train_features, train_scores = self.bind_train(train_loader.dataset, seed)
        val_features, val_scores = self.bind_val(val_loader.dataset, seed + 1)
        train_size = min(size, len(train_loader.dataset))
        val_size = min(max(size // 4, 1), len(val_loader.dataset))
        train_idx = self.select_indices(
            dataset_labels(train_loader.dataset),
            train_size,
            seed,
            features=train_features,
            scores=train_scores,
        )
        val_idx = self.select_indices(
            dataset_labels(val_loader.dataset),
            val_size,
            seed + 1,
            features=val_features,
            scores=val_scores,
        )
        sim_train = DataLoader(
            Subset(train_loader.dataset, train_idx),
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=DATALOADER_NUM_WORKERS,
        )
        sim_val = DataLoader(
            Subset(val_loader.dataset, val_idx),
            batch_size=val_loader.batch_size,
            num_workers=DATALOADER_NUM_WORKERS,
        )
        return sim_train, sim_val
