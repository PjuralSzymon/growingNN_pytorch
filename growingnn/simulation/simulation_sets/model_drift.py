"""
Model-drift-triggered subset refresh.

Proposed method — no direct reference. GrowingNN-specific refresh that rebuilds
an inner simulation set when current-model embeddings on a fixed anchor set drift.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.protected import ProtectedSimulationSet
from growingnn.simulation.simulation_sets.commons import (
    dataset_labels,
    extract_embeddings,
    protected_sampling_indices,
    require_model,
    scoring_loader,
)


def mean_cosine_distance(old_embeddings: torch.Tensor, new_embeddings: torch.Tensor) -> float:
    old_norm = torch.linalg.vector_norm(old_embeddings, dim=1)
    new_norm = torch.linalg.vector_norm(new_embeddings, dim=1)
    both_zero = (old_norm < 1e-8) & (new_norm < 1e-8)
    old_embeddings = F.normalize(old_embeddings, p=2, dim=1)
    new_embeddings = F.normalize(new_embeddings, p=2, dim=1)
    similarities = (old_embeddings * new_embeddings).sum(dim=1)
    similarities = torch.where(both_zero, torch.ones_like(similarities), similarities)
    similarities = torch.nan_to_num(similarities, nan=0.0)
    distances = 1.0 - similarities
    return float(distances.mean().item())


class ModelDriftSimulationSet(SimulationSet):
    """
    2. Model-Drift-Triggered Subset Refresh.

    Paper: none. GrowingNN-specific refresh policy, not a sample picker.

    Rebuild the inner selector when current-model embeddings on a fixed anchor set drift.

    Pseudocode:
        embed = model embeddings on fixed 256-example anchor
        if mean cosine distance(embed, last_embed) >= 0.1:
            sim_set = selector.generate(...)
            last_embed = embed
        else:
            keep cached sim_set
    """

    def __init__(
        self,
        selector: SimulationSet | None = None,
        anchor_size: int = 256,
        drift_threshold: float = 0.1,
    ) -> None:
        self.selector = selector if selector is not None else ProtectedSimulationSet()
        self.anchor_size = anchor_size
        self.drift_threshold = drift_threshold
        self._cached: tuple[DataLoader, DataLoader] | None = None
        self._reference_embeddings: torch.Tensor | None = None
        self._anchor_indices: list[int] | None = None

    def _anchor_loader(self, train_loader: DataLoader, seed: int) -> DataLoader:
        if self._anchor_indices is None:
            labels = dataset_labels(train_loader.dataset)
            self._anchor_indices = protected_sampling_indices(
                labels, min(self.anchor_size, len(train_loader.dataset)), seed,
            )
        return scoring_loader(Subset(train_loader.dataset, self._anchor_indices), batch_size=32)

    def _current_embeddings(self, model, train_loader: DataLoader, seed: int) -> torch.Tensor:
        return extract_embeddings(model, self._anchor_loader(train_loader, seed))

    def needs_refresh(self, model) -> bool:
        if self._cached is None or self._reference_embeddings is None:
            return True
        train_loader = getattr(self, "_last_train_loader", None)
        seed = getattr(self, "_last_seed", 0)
        if train_loader is None:
            return True
        current = self._current_embeddings(model, train_loader, seed)
        if current.shape != self._reference_embeddings.shape:
            return True
        return mean_cosine_distance(self._reference_embeddings, current) >= self.drift_threshold

    def generate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int = 0,
        model=None,
    ) -> tuple[DataLoader, DataLoader]:
        self._last_train_loader = train_loader
        self._last_seed = seed
        model = require_model(model)
        current = self._current_embeddings(model, train_loader, seed)
        if (
            self._cached is not None
            and self._reference_embeddings is not None
            and current.shape == self._reference_embeddings.shape
            and mean_cosine_distance(self._reference_embeddings, current) < self.drift_threshold
        ):
            return self._cached
        self._cached = self.selector.generate(train_loader, val_loader, size, seed, model)
        self._reference_embeddings = current.detach().clone()
        return self._cached
