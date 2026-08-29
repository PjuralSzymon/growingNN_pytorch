"""Abstract simulation-set generator."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class SimulationSet(ABC):
    """Build the small train/val loaders used by simulation scoring."""

    def needs_refresh(self, model) -> bool:
        return False

    @abstractmethod
    def generate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int = 0,
        model=None,
    ) -> tuple[DataLoader, DataLoader]:
        raise NotImplementedError
