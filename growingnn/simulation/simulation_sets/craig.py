"""
CRAIG weighted coreset simulation-set selection.

Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020). Coresets for Data-efficient
Training of Machine Learning Models. ICML 2020, PMLR 119, 6950-6960.
BibTeX: pmlr-v119-mirzasoleiman20a
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.commons import (
    class_quota,
    collect_last_layer_grads,
    dataset_labels,
    ground_set_indices,
    indices_to_loaders,
    require_model,
)


def craig_select(gradients: torch.Tensor, count: int) -> tuple[list[int], torch.Tensor]:
    n = gradients.shape[0]
    count = min(count, n)
    if count <= 0:
        return [], torch.ones(0)
    distances = torch.cdist(gradients.float(), gradients.float())
    selected = [int(distances.sum(dim=0).argmin().item())]
    min_dist = distances[selected[0]].clone()
    for _ in range(count - 1):
        new_mins = torch.minimum(min_dist.unsqueeze(0), distances)
        improvements = (min_dist.unsqueeze(0) - new_mins).clamp_min(0).sum(dim=1)
        improvements[selected] = CraigSimulationSet.ALREADY_SELECTED_IMPROVEMENT
        nxt = int(improvements.argmax().item())
        selected.append(nxt)
        min_dist = torch.minimum(min_dist, distances[nxt])
    nearest = distances[:, selected].argmin(dim=1)
    weights = torch.zeros(len(selected))
    for assignment in nearest.tolist():
        weights[assignment] += 1.0
    return selected, weights.clamp_min(1.0)


class CraigSimulationSet(SimulationSet):
    """
    8. CRAIG.

    Paper: Mirzasoleiman, Bilmes, Leskovec 2020, ICML, PMLR 119, 6950-6960
    (pmlr-v119-mirzasoleiman20a).

    Cover last-layer gradient space with a small weighted coreset (facility location).

    Pseudocode:
        g_i = last-layer grad of sample i
        S = {argmin_j sum_i ||g_i - g_j||}
        while |S| < K:
            add the point that most reduces sum_i min_{j in S} ||g_i - g_j||
        weight_j = how many points assign to j
    """

    ALREADY_SELECTED_IMPROVEMENT = float("-inf")

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
        candidates = ground_set_indices(labels, size, seed)
        grads, candidate_labels = collect_last_layer_grads(
            model, train_loader.dataset, candidates, train_loader.batch_size or 32,
        )
        unique_classes = torch.unique(candidate_labels).tolist()
        quota = class_quota(size, len(unique_classes))
        selected_local: list[int] = []
        weights = torch.ones(0)
        for class_id, n_to_select in zip(unique_classes, quota):
            mask = candidate_labels == class_id
            local_pos = torch.nonzero(mask, as_tuple=False).view(-1)
            if local_pos.numel() == 0 or n_to_select <= 0:
                continue
            picks, class_weights = craig_select(grads[local_pos], n_to_select)
            selected_local.extend(int(local_pos[i]) for i in picks)
            weights = torch.cat([weights, class_weights], dim=0)
        train_idx = [candidates[i] for i in selected_local]
        return indices_to_loaders(train_loader, val_loader, train_idx, train_weights=weights)
