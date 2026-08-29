"""
Moderate-difficulty sampling for GrowingNN simulation sets.

Related motivation only, not the exact algorithm:
Paul, M., Ganguli, S., & Dziugaite, G. K. (2021). Deep Learning on a Data Diet:
Finding Important Examples Early in Training. NeurIPS 34, 20596-20607.
BibTeX: paul2021dataDiet
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.commons import (
    class_quota,
    dataset_labels,
    evenly_spaced_select,
    indices_to_loaders,
    require_model,
    scoring_loader,
)


class ModerateDifficultySimulationSet(SimulationSet):
    """
    1. Moderate-Difficulty Sampling.

    Paper: none for this exact method. Related only: Paul et al. 2021, Data Diet
    (paul2021dataDiet).

    Keep per-class examples with medium cross-entropy; drop the easiest and hardest.

    Pseudocode:
        for each class:
            score = CE(model(x), y)
            keep middle quantile [0.25, 0.75)
            take evenly spaced K/C samples
    """

    def __init__(self, lower_quantile: float = 0.25, upper_quantile: float = 0.75) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def needs_refresh(self, model) -> bool:
        return True

    def generate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int = 0,
        model=None,
    ) -> tuple[DataLoader, DataLoader]:
        model = require_model(model)
        labels = dataset_labels(train_loader.dataset)
        num_classes = int(torch.unique(labels).numel())
        device = next(model.parameters()).device
        scored: list[tuple[int, int, float]] = []
        offset = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in scoring_loader(train_loader.dataset, train_loader.batch_size or 32):
                    x, y = batch[0].to(device), batch[1].to(device)
                    losses = F.cross_entropy(model(x), y, reduction="none")
                    for label, loss in zip(y.tolist(), losses.detach().cpu().tolist()):
                        scored.append((offset, int(label), float(loss)))
                        offset += 1
        finally:
            model.train(was_training)

        selected: list[int] = []
        for class_id, n_to_select in enumerate(class_quota(size, num_classes)):
            class_items = [item for item in scored if item[1] == class_id]
            class_items.sort(key=lambda item: item[2])
            n = len(class_items)
            start = int(self.lower_quantile * n)
            end = max(start + 1, int(self.upper_quantile * n)) if n else 0
            candidates = class_items[start:end] or class_items
            selected.extend(item[0] for item in evenly_spaced_select(candidates, n_to_select))
        return indices_to_loaders(train_loader, val_loader, selected)
