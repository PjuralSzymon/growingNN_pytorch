"""
k-Center greedy / CoreSet simulation-set selection.

Sener, O., & Savarese, S. (2018). Active Learning for Convolutional Neural
Networks: A Core-Set Approach. ICLR 2018.
BibTeX: sener2018coreSet
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.commons import (
    class_quota,
    dataset_labels,
    extract_embeddings,
    indices_to_loaders,
    require_model,
    scoring_loader,
)


def _kcenter_indices(embeddings: torch.Tensor, count: int, seed: int) -> list[int]:
    n = embeddings.shape[0]
    if count <= 0 or n == 0:
        return []
    count = min(count, n)
    centroid = embeddings.mean(dim=0, keepdim=True)
    selected = [int(torch.norm(embeddings - centroid, dim=1).argmin().item())]
    while len(selected) < count:
        distances = torch.cdist(embeddings, embeddings[selected])
        next_idx = int(distances.min(dim=1).values.argmax().item())
        if next_idx in selected:
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            generator = torch.Generator().manual_seed(seed + len(selected))
            next_idx = remaining[int(torch.randint(len(remaining), (1,), generator=generator).item())]
        selected.append(next_idx)
    return selected


class KCenterSimulationSet(SimulationSet):
    """
    5. k-Center Greedy / CoreSet.

    Paper: Sener and Savarese 2018, ICLR (sener2018coreSet).

    Cover current-model embedding space by always adding the point farthest from the set.

    Pseudocode:
        embed = model embeddings of train
        pick first = nearest to mean embed
        while |S| < K:
            add argmax_i min_{j in S} ||embed_i - embed_j||
    """

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
        embeddings = extract_embeddings(model, scoring_loader(train_loader.dataset, train_loader.batch_size or 32))
        unique_classes = torch.unique(labels).tolist()
        quota = class_quota(size, len(unique_classes))
        selected: list[int] = []
        for class_id, n_to_select in zip(unique_classes, quota):
            class_idx = torch.nonzero(labels == class_id, as_tuple=False).view(-1)
            local = _kcenter_indices(embeddings[class_idx], n_to_select, seed + int(class_id))
            selected.extend(int(class_idx[i]) for i in local)
        return indices_to_loaders(train_loader, val_loader, selected)
