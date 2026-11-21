"""EL2N simulation set: keep the largest early prediction errors."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from growingnn.simulation.simulation_sets.base import (
    SimulationSetAlgorithm,
    dataset_labels,
    dataset_matrix,
)


class El2nSimulationSet(SimulationSetAlgorithm):
    """Score examples by the early squared error of a linear probe and keep the largest."""

    name = "el2n"

    def __init__(self, epochs: int = 10) -> None:
        self.epochs = max(1, int(epochs))
        self._model: nn.Linear | None = None
        self._class_to_index: dict[int, int] = {}

    def bind_train(
        self,
        dataset: Dataset,
        seed: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        features = dataset_matrix(dataset)
        labels = dataset_labels(dataset).reshape(-1).cpu()
        self._model, self._class_to_index, scores = _fit_el2n(
            features,
            labels,
            epochs=self.epochs,
            seed=seed,
        )
        return features, scores

    def bind_val(
        self,
        dataset: Dataset,
        seed: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._model is None:
            return self.bind_train(dataset, seed)
        features = dataset_matrix(dataset)
        labels = dataset_labels(dataset).reshape(-1).cpu()
        return features, _score_el2n(self._model, features, labels, self._class_to_index)

    def _select_class(
        self,
        indices: torch.Tensor,
        quota: int,
        seed: int,
        features: torch.Tensor | None,
        scores: torch.Tensor | None,
    ) -> list[int]:
        if scores is None:
            raise ValueError("EL2N selection requires a score for every example")
        order = sorted(
            range(len(indices)),
            key=lambda row: (-float(scores[row]), int(indices[row])),
        )
        return [int(indices[row]) for row in order[:quota]]


def _fit_el2n(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    seed: int,
) -> tuple[nn.Linear, dict[int, int], torch.Tensor]:
    """Train a linear probe and average the per-example squared error over epochs."""
    values = features.detach().float().cpu()
    targets = labels.reshape(-1).cpu()
    classes = torch.unique(targets)
    class_to_index = {int(label): index for index, label in enumerate(classes.tolist())}
    local_targets = torch.tensor([class_to_index[int(label)] for label in targets.tolist()])
    generator = torch.Generator().manual_seed(seed)
    model = nn.Linear(values.shape[1], len(classes))
    with torch.no_grad():
        weight = torch.randn(len(classes), values.shape[1], generator=generator)
        model.weight.copy_(weight / values.shape[1] ** 0.5)
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    running = torch.zeros(values.shape[0])
    batch_size = min(128, values.shape[0])
    for _epoch in range(epochs):
        permutation = torch.randperm(values.shape[0], generator=generator)
        model.train()
        for start in range(0, values.shape[0], batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(values[batch]), local_targets[batch])
            loss.backward()
            optimizer.step()
        running += _score_rows(model, values, local_targets)
    return model, class_to_index, running / epochs


def _score_el2n(
    model: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_to_index: dict[int, int],
) -> torch.Tensor:
    """Score a new split with one forward pass of a fitted probe."""
    values = features.detach().float().cpu()
    local_targets = []
    for label in labels.reshape(-1).tolist():
        mapped = class_to_index.get(int(label))
        local_targets.append(0 if mapped is None else mapped)
    targets = torch.tensor(local_targets)
    return _score_rows(model, values, targets)


def _score_rows(
    model: nn.Linear,
    values: torch.Tensor,
    local_targets: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(values), dim=1)
        one_hot = torch.nn.functional.one_hot(local_targets, probabilities.shape[1]).float()
        return (probabilities - one_hot).pow(2).sum(dim=1)
