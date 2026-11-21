"""Class-balanced random simulation set. This is the current sampler."""

from __future__ import annotations

import torch

from growingnn.core.logger import logger
from growingnn.simulation.simulation_sets.base import SimulationSetAlgorithm


class StratifiedRandomSimulationSet(SimulationSetAlgorithm):
    """Keep an equal random count from every class."""

    name = "stratified_random"

    def select_indices(
        self,
        labels: torch.Tensor,
        size: int,
        seed: int = 0,
        *,
        features: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
    ) -> list[int]:
        """Draw the same stratified subset the training loop has been using."""
        labels = torch.as_tensor(labels).reshape(-1)
        generator = torch.Generator().manual_seed(seed)
        unique_classes = torch.unique(labels, dim=0)
        samples_per_class = max(1, size // len(unique_classes))
        selected: list[int] = []
        for cls in unique_classes.tolist():
            class_indices = torch.nonzero(labels == cls, as_tuple=False).view(-1)
            quota = min(samples_per_class, len(class_indices))
            pick = class_indices[torch.randperm(len(class_indices), generator=generator)[:quota]]
            selected.extend(pick.tolist())
        if len(selected) > size:
            logger.info(
                "Simulation sample count %s exceeds requested %s (classes=%s, min one per class)",
                len(selected),
                size,
                len(unique_classes),
            )
        return selected

    def _select_class(
        self,
        indices: torch.Tensor,
        quota: int,
        seed: int,
        features: torch.Tensor | None,
        scores: torch.Tensor | None,
    ) -> list[int]:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(indices), generator=generator)[:quota]
        return indices[order].tolist()
