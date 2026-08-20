"""
HCDC-inspired synthetic simulation set.

Ding, M., Xu, Y., Rabbani, T., Liu, X., Gravelle, B., Ranadive, T., Tuan, T.-C.,
& Huang, F. (2024). Calibrated Dataset Condensation for Faster Hyperparameter
Search. arXiv:2405.17535.
BibTeX: ding2024calibrated

This GrowingNN version is an explicit simplification of the paper: it matches
last-layer validation gradients of the current model between real val data and
synthetic images. It does not run implicit-function hypergradients over a
discrete architecture variable, and it does not train with Spearman/Kendall.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from growingnn.core.config import DATALOADER_NUM_WORKERS
from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.commons import (
    dataset_labels,
    mean_last_layer_grad,
    per_example_last_layer_grads,
    protected_sampling_indices,
    require_model,
)


class HcdcSimulationSet(SimulationSet):
    """
    4. HCDC.

    Paper: Ding et al. 2024, arXiv:2405.17535 (ding2024calibrated). This version is a
    simplification: last-layer val-gradient match, not full IFT hypergradients.

    Learn a small synthetic set so its last-layer val gradient matches real val data.

    Pseudocode:
        init synthetic x from a protected sample
        for a few steps:
            g_syn = last-layer grad on synthetic
            g_val = last-layer grad on real val
            x := x - lr * d||g_syn - g_val|| / dx
        return TensorDataset(x, y)
    """

    def __init__(
        self,
        steps: int = 8,
        synthetic_lr: float = 0.1,
        time_cap: float = 20.0,
    ) -> None:
        self.steps = steps
        self.synthetic_lr = synthetic_lr
        self.time_cap = time_cap
        self._built = False

    def needs_refresh(self, model) -> bool:
        return not self._built

    def generate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int = 0,
        model=None,
    ) -> tuple[DataLoader, DataLoader]:
        model = require_model(model)
        loaders = self._condense(train_loader, val_loader, size, seed, model)
        self._built = True
        return loaders

    def _condense(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        size: int,
        seed: int,
        model,
    ) -> tuple[DataLoader, DataLoader]:
        probe_x, probe_y = next(iter(train_loader))
        labels = dataset_labels(train_loader.dataset)
        init_idx = protected_sampling_indices(labels, size, seed)
        init_x = torch.stack([train_loader.dataset[i][0] for i in init_idx]).clone()
        init_y = torch.tensor([int(train_loader.dataset[i][1]) for i in init_idx], dtype=torch.long)
        if init_x.shape[0] != size:
            extra = size - init_x.shape[0]
            init_x = torch.cat([init_x, probe_x[:extra].cpu()], dim=0) if extra > 0 else init_x[:size]
            init_y = torch.cat([init_y, probe_y[:extra].cpu().long()], dim=0) if extra > 0 else init_y[:size]
        synthetic_x = init_x.detach().clone().requires_grad_(True)
        synthetic_y = init_y.detach().clone()
        optimizer = torch.optim.Adam([synthetic_x], lr=self.synthetic_lr)
        device = next(model.parameters()).device
        deadline = time.time() + self.time_cap
        h_real = mean_last_layer_grad(model, val_loader).detach().to(device)
        was_training = model.training
        requires_grad = [p.requires_grad for p in model.parameters()]
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        try:
            for _ in range(self.steps):
                if time.time() >= deadline:
                    break
                optimizer.zero_grad()
                synth_x = synthetic_x.to(device)
                synth_y = synthetic_y.to(device)
                h_synthetic = per_example_last_layer_grads(model, synth_x, synth_y).mean(0)
                matching_loss = (h_real - h_synthetic).pow(2).sum()
                matching_loss.backward()
                optimizer.step()
        finally:
            for param, required in zip(model.parameters(), requires_grad):
                param.requires_grad_(required)
            model.train(was_training)
        train_ds = TensorDataset(synthetic_x.detach().cpu(), synthetic_y.cpu())
        val_size = min(max(size // 4, 1), len(val_loader.dataset))
        val_idx = protected_sampling_indices(dataset_labels(val_loader.dataset), val_size, seed + 1)
        val_x = torch.stack([val_loader.dataset[i][0] for i in val_idx])
        val_y = torch.tensor([int(val_loader.dataset[i][1]) for i in val_idx], dtype=torch.long)
        val_ds = TensorDataset(val_x, val_y)
        sim_train = DataLoader(
            train_ds,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=DATALOADER_NUM_WORKERS,
        )
        sim_val = DataLoader(
            val_ds,
            batch_size=val_loader.batch_size,
            num_workers=DATALOADER_NUM_WORKERS,
        )
        return sim_train, sim_val
