"""Coverage-centric simulation set: spread a budget across difficulty strata."""

from __future__ import annotations

import torch

from growingnn.simulation.simulation_sets.base import SimulationSetAlgorithm
from growingnn.simulation.simulation_sets.el2n import El2nSimulationSet


class CcsSimulationSet(SimulationSetAlgorithm):
    """Drop the hardest tail, then sample every remaining difficulty band."""

    name = "ccs"

    def __init__(
        self,
        strata: int = 50,
        hard_cutoff: float = 0.30,
        epochs: int = 10,
    ) -> None:
        self.strata = max(1, int(strata))
        self.hard_cutoff = min(max(float(hard_cutoff), 0.0), 1.0)
        self._difficulty = El2nSimulationSet(epochs=epochs)

    def bind_train(self, dataset, seed: int):
        return self._difficulty.bind_train(dataset, seed)

    def bind_val(self, dataset, seed: int):
        return self._difficulty.bind_val(dataset, seed)

    def _select_class(
        self,
        indices: torch.Tensor,
        quota: int,
        seed: int,
        features: torch.Tensor | None,
        scores: torch.Tensor | None,
    ) -> list[int]:
        if scores is None:
            raise ValueError("CCS selection requires a difficulty score for every example")
        ranked = sorted(
            range(len(indices)),
            key=lambda row: (-float(scores[row]), int(indices[row])),
        )
        drop = min(int(len(indices) * self.hard_cutoff), max(0, len(indices) - quota))
        kept = ranked[drop:]
        bins = _difficulty_bins(
            [int(indices[row]) for row in kept],
            [float(scores[row]) for row in kept],
            self.strata,
        )
        return _sample_bins(bins, quota, seed)


def _difficulty_bins(
    indices: list[int],
    scores: list[float],
    strata: int,
) -> list[list[int]]:
    if not indices:
        return []
    low = min(scores)
    high = max(scores)
    if high == low or strata <= 1:
        return [list(indices)]
    width = (high - low) / strata
    bins: list[list[int]] = [[] for _ in range(strata)]
    for index, score in zip(indices, scores):
        slot = int((score - low) / width)
        bins[min(slot, strata - 1)].append(index)
    return [slot for slot in bins if slot]


def _sample_bins(bins: list[list[int]], quota: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    budget = quota
    remaining = [list(slot) for slot in bins]
    while remaining and budget > 0:
        smallest = min(range(len(remaining)), key=lambda slot: (len(remaining[slot]), min(remaining[slot])))
        share = budget // len(remaining)
        if share == 0:
            share = 1
        share = min(share, len(remaining[smallest]), budget)
        pool = remaining.pop(smallest)
        order = torch.randperm(len(pool), generator=generator).tolist()
        selected.extend(pool[index] for index in order[:share])
        budget -= share
    return selected
