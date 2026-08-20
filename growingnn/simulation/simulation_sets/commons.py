"""Shared helpers for simulation-set generators."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from growingnn.core.config import DATALOADER_NUM_WORKERS
from growingnn.core.logger import logger


def require_model(model):
    if model is None:
        raise ValueError("generate requires a model")
    return model


def dataset_labels(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets)
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return dataset.tensors[1]
    labels: list[Any] = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(item[1] if isinstance(item, tuple) else item[-1])
    return torch.as_tensor(labels)


def protected_sampling_indices(
    labels: torch.Tensor | np.ndarray,
    n: int,
    seed: int = 0,
) -> list[int]:
    labels = torch.as_tensor(labels).reshape(-1)
    generator = torch.Generator().manual_seed(seed)
    unique_classes = torch.unique(labels, dim=0)
    samples_per_class = max(1, n // len(unique_classes))
    selected: list[int] = []
    for cls in unique_classes.tolist():
        class_indices = torch.nonzero(labels == cls, as_tuple=False).view(-1)
        k = min(samples_per_class, len(class_indices))
        pick = class_indices[torch.randperm(len(class_indices), generator=generator)[:k]]
        selected.extend(pick.tolist())
    if len(selected) > n:
        logger.info(
            "Simulation sample count %s exceeds requested %s (classes=%s, min one per class)",
            len(selected), n, len(unique_classes),
        )
    return selected


def class_quota(size: int, n_classes: int) -> list[int]:
    if n_classes <= 0:
        return []
    base = size // n_classes
    remainder = size % n_classes
    return [base + (1 if i < remainder else 0) for i in range(n_classes)]


def evenly_spaced_select(sorted_candidates: Sequence[Any], n_to_select: int) -> list[Any]:
    if n_to_select >= len(sorted_candidates):
        return list(sorted_candidates)
    if n_to_select <= 0 or not sorted_candidates:
        return []
    positions = torch.linspace(0, len(sorted_candidates) - 1, steps=n_to_select)
    return [sorted_candidates[int(pos)] for pos in positions.round().long().tolist()]


def class_balanced_top_scores(
    scored_samples: list[tuple[int, int, float]],
    size: int,
    num_classes: int,
    reverse: bool = True,
) -> list[int]:
    selected: list[int] = []
    for class_id, n_to_select in enumerate(class_quota(size, num_classes)):
        class_items = [item for item in scored_samples if item[1] == class_id]
        class_items.sort(key=lambda item: item[2], reverse=reverse)
        selected.extend(item[0] for item in class_items[:n_to_select])
    return selected


def ground_set_indices(labels: torch.Tensor, size: int, seed: int, multiplier: int = 4) -> list[int]:
    cap = min(len(labels), max(size * multiplier, size))
    return protected_sampling_indices(labels, cap, seed)


def last_linear_module(model: nn.Module) -> nn.Linear:
    last: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        raise ValueError("Simulation set helpers need an nn.Linear classifier")
    return last


def last_layer_parameters(model: nn.Module) -> list[nn.Parameter]:
    linear = last_linear_module(model)
    return [linear.weight, linear.bias] if linear.bias is not None else [linear.weight]


def _module_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_embeddings(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    linear = last_linear_module(model)
    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        features = inputs[0].detach()
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.ndim > 2:
            features = features.flatten(1)
        captured.append(features.cpu())

    handle = linear.register_forward_hook(_hook)
    was_training = model.training
    model.eval()
    device = _module_device(model)
    try:
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                model(x)
    finally:
        handle.remove()
        model.train(was_training)
    if not captured:
        return torch.empty(0, 0)
    return torch.cat(captured, dim=0)


def per_example_last_layer_grads(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    linear = last_linear_module(model)
    features: list[torch.Tensor] = []

    def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
        feat = inputs[0]
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.ndim > 2:
            feat = feat.flatten(1)
        features.append(feat)

    handle = linear.register_forward_hook(_hook)
    was_training = model.training
    model.eval()
    try:
        logits = model(x)
        feat = features[0]
        residual = torch.softmax(logits, dim=1) - F.one_hot(y, num_classes=logits.shape[1]).float()
        grad_weight = residual.unsqueeze(2) * feat.unsqueeze(1)
        parts = [grad_weight.flatten(1)]
        if linear.bias is not None:
            parts.append(residual)
        return torch.cat(parts, dim=1)
    finally:
        handle.remove()
        model.train(was_training)


def mean_last_layer_grad(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    device = _module_device(model)
    total: torch.Tensor | None = None
    count = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        grads = per_example_last_layer_grads(model, x, y)
        total = grads.sum(0) if total is None else total + grads.sum(0)
        count += grads.shape[0]
    if total is None or count == 0:
        raise ValueError("Cannot compute a reference gradient from an empty loader")
    return total / count


def collect_last_layer_grads(
    model: nn.Module,
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _module_device(model)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        rows.append(per_example_last_layer_grads(model, x, y).detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(rows, dim=0), torch.cat(labels, dim=0)


def scoring_loader(dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=DATALOADER_NUM_WORKERS)


def indices_to_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_idx: list[int],
    val_idx: list[int] | None = None,
    *,
    train_weights: torch.Tensor | None = None,
) -> tuple[DataLoader, DataLoader]:
    if val_idx is None:
        val_labels = dataset_labels(val_loader.dataset)
        val_size = min(max(len(train_idx) // 4, 1), len(val_loader.dataset))
        val_idx = protected_sampling_indices(val_labels, val_size, seed=1)
    train_kwargs: dict[str, Any] = {
        "batch_size": train_loader.batch_size,
        "num_workers": DATALOADER_NUM_WORKERS,
    }
    if train_weights is None:
        train_kwargs["shuffle"] = True
    else:
        weights = train_weights.detach().cpu().float().clamp_min(1e-8)
        train_kwargs["sampler"] = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
        )
    sim_train = DataLoader(Subset(train_loader.dataset, train_idx), **train_kwargs)
    sim_val = DataLoader(
        Subset(val_loader.dataset, val_idx),
        batch_size=val_loader.batch_size,
        num_workers=DATALOADER_NUM_WORKERS,
    )
    return sim_train, sim_val


def subset_indices(loader: DataLoader) -> list[int]:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        return list(dataset.indices)
    return list(range(len(dataset)))
