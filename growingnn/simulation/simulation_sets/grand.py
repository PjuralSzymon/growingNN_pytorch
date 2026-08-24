"""
GraNd simulation-set selection.

Paul, M., Ganguli, S., & Dziugaite, G. K. (2021). Deep Learning on a Data Diet:
Finding Important Examples Early in Training. NeurIPS 34, 20596-20607.
BibTeX: paul2021dataDiet
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.commons import (
    class_balanced_top_scores,
    dataset_labels,
    indices_to_loaders,
    per_example_last_layer_grads,
    require_model,
    scoring_loader,
)


class GrandSimulationSet(SimulationSet):
    """
    6. GraNd.

    Paper: Paul et al. 2021, NeurIPS 34, 20596-20607 (paul2021dataDiet).

    Keep class-balanced examples with the largest last-layer gradient norm.

    Pseudocode:
        for each sample:
            GraNd = || d CE(model(x), y) / d last_layer ||
        per class keep the top K/C scores
    """

    def __init__(self, selection_mode: str = "highest") -> None:
        self.selection_mode = selection_mode

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
        for batch in scoring_loader(train_loader.dataset, train_loader.batch_size or 32):
            x, y = batch[0].to(device), batch[1].to(device)
            grads = per_example_last_layer_grads(model, x, y)
            norms = torch.linalg.vector_norm(grads, dim=1)
            for label, score in zip(y.tolist(), norms.detach().cpu().tolist()):
                scored.append((offset, int(label), float(score)))
                offset += 1
        reverse = self.selection_mode != "lowest"
        selected = class_balanced_top_scores(scored, size, num_classes, reverse=reverse)
        return indices_to_loaders(train_loader, val_loader, selected)
