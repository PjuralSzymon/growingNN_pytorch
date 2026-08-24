"""
GRAD-MATCH simulation-set selection.

Killamsetty, K., S, D., Ramakrishnan, G., De, A., & Iyer, R. (2021). GRAD-MATCH:
Gradient Matching based Data Subset Selection for Efficient Deep Model Training.
ICML 2021, PMLR 139, 5464-5474.
BibTeX: pmlr-v139-killamsetty21a
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
    mean_last_layer_grad,
    require_model,
)


def omp_select(gradients: torch.Tensor, g_reference: torch.Tensor, count: int) -> tuple[list[int], torch.Tensor]:
    n = gradients.shape[0]
    count = min(count, n)
    residual = g_reference.reshape(-1).cpu().float()
    matrix = gradients.cpu().float()
    unused = list(range(n))
    selected: list[int] = []
    weights = torch.ones(count)
    for step in range(count):
        if not unused:
            break
        dots = (matrix[unused] @ residual).abs()
        pick = unused.pop(int(dots.argmax().item()))
        selected.append(pick)
        design = matrix[selected].T
        solution = torch.linalg.lstsq(design, residual.unsqueeze(1), rcond=None).solution.view(-1)
        weights[: len(selected)] = solution
        residual = residual - design @ solution
    return selected, weights[: len(selected)]


class GradMatchSimulationSet(SimulationSet):
    """
    3. GRAD-MATCH.

    Paper: Killamsetty et al. 2021, ICML, PMLR 139, 5464-5474 (pmlr-v139-killamsetty21a).

    Pick a weighted subset whose last-layer gradient matches the full-data gradient.

    Pseudocode:
        g_full = mean last-layer grad over train
        for each class:
            pick K/C points by OMP so weighted grads ~ g_full
        sample with those weights
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
        candidates = ground_set_indices(labels, size, seed)
        g_reference = mean_last_layer_grad(model, train_loader).detach().cpu()
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
            picks, class_weights = omp_select(grads[local_pos], g_reference, n_to_select)
            selected_local.extend(int(local_pos[i]) for i in picks)
            weights = torch.cat([weights, class_weights], dim=0)
        train_idx = [candidates[i] for i in selected_local]
        return indices_to_loaders(train_loader, val_loader, train_idx, train_weights=weights)
