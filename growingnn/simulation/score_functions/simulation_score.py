"""Weighted composite score for simulation rollouts."""

from __future__ import annotations

import copy

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.score_by_learning import score_acc, score_loss
from growingnn.simulation.score_functions.score_efficiency import score_count_weights, score_time


class SimulationScore:
    _SCORE_FUNCTIONS = {
        "weight_acc": score_acc,
        "weight_loss": score_loss,
        "weight_time": score_time,
        "weight_countW": score_count_weights,
    }

    def __init__(
        self,
        weight_acc: float = 1.0,
        weight_loss: float = 0.0,
        weight_time: float = 0.0,
        weight_countW: float = 0.5,
    ):
        self.weights = {
            "weight_acc": weight_acc,
            "weight_loss": weight_loss,
            "weight_time": weight_time,
            "weight_countW": weight_countW,
        }

    def weight_sum(self) -> float:
        return sum(self.weights.values())

    def score(
        self,
        model: nn.Module | fx.GraphModule,
        running_config: RunningConfig,
    ) -> float:
        total = 0.0
        for key, fn in self._SCORE_FUNCTIONS.items():
            weight = self.weights[key]
            if weight > 0.0:
                total += weight * fn(copy.deepcopy(model), running_config)
        return total / self.weight_sum()
