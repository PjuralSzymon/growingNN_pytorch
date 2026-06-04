"""Weighted composite score for simulation rollouts."""

from __future__ import annotations

import copy

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
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
        board = running_config.experiment_board
        board_metrics = board.simulation_metrics if board is not None else None
        if board_metrics is not None:
            board_metrics.clear()
        for key, fn in self._SCORE_FUNCTIONS.items():
            weight = self.weights[key]
            if weight <= 0.0:
                continue
            term_score = fn(copy.deepcopy(model), running_config)
            if board_metrics is not None:
                board_metrics[f"{key}_score"] = term_score
                board_metrics[f"{key}_weight"] = weight
                board_metrics[f"{key}_weighted"] = weight * term_score
            total += weight * term_score
        divisor = self.weight_sum()
        if divisor == 0.0:
            logger.error("SimulationScore.score: all weights are 0, returning 0 instead of NaN")
            return 0.0
        composite = total / divisor
        if board_metrics is not None:
            board_metrics["composite_score"] = composite
        return composite
