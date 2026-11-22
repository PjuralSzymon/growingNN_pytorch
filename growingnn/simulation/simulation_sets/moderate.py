"""Moderate simulation set: keep examples near the median distance to the class center."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from growingnn.simulation.simulation_sets.base import (
    SimulationSetAlgorithm,
    class_center_distance,
    dataset_matrix,
)


class ModerateSimulationSet(SimulationSetAlgorithm):
    """Select the examples whose distance to the class center is closest to the median."""

    name = "moderate"

    def bind_train(
        self,
        dataset: Dataset,
        seed: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return dataset_matrix(dataset), None

    def _select_class(
        self,
        indices: torch.Tensor,
        quota: int,
        seed: int,
        features: torch.Tensor | None,
        scores: torch.Tensor | None,
    ) -> list[int]:
        if features is None:
            raise ValueError("Moderate selection requires one feature vector per example")
        _center, distances, median = class_center_distance(features)
        closeness = (distances - median).abs()
        order = sorted(
            range(len(indices)),
            key=lambda row: (float(closeness[row]), int(indices[row])),
        )
        return [int(indices[row]) for row in order[:quota]]
