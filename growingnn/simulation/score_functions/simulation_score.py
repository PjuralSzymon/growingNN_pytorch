"""Weighted composite score for simulation rollouts."""

from __future__ import annotations

import copy
import math

import torch.fx as fx
import torch.nn as nn

from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
from growingnn.simulation.score_functions.score_by_learning import (
    AccuracyMetric,
    parse_accuracy_metric,
    score_acc,
    score_loss,
)
from growingnn.simulation.score_functions.score_efficiency import score_count_weights, score_time


class SimulationScore:
    _SCORE_FUNCTIONS = {
        "weight_acc": score_acc,
        "weight_loss": score_loss,
        "weight_time": score_time,
        "weight_countW": score_count_weights,
    }
    _TRAINING_TERMS = ("weight_acc", "weight_loss", "weight_time")

    def __init__(
        self,
        weight_acc: float = 1.0,
        weight_loss: float = 0.0,
        weight_time: float = 0.0,
        weight_countW: float = 0.5,
        accuracy_metric: AccuracyMetric | str = AccuracyMetric.VAL_ACC,
        average_last_epochs: int = 1,
    ):
        self.weights = {
            "weight_acc": weight_acc,
            "weight_loss": weight_loss,
            "weight_time": weight_time,
            "weight_countW": weight_countW,
        }
        self.accuracy_metric = parse_accuracy_metric(accuracy_metric)
        # Averaging the last few scoring epochs smooths epoch-to-epoch validation noise.
        self.average_last_epochs = int(average_last_epochs)

    def weight_sum(self) -> float:
        return sum(self.weights.values())

    def score(
        self,
        model: nn.Module | fx.GraphModule,
        running_config: RunningConfig,
    ) -> float:
        return self.score_and_train(model, running_config)[0]

    def score_and_train(
        self,
        model: nn.Module | fx.GraphModule,
        running_config: RunningConfig,
    ) -> tuple[float, nn.Module | fx.GraphModule]:
        """Score a copy of *model* and also return that copy after its scoring training.

        The first term that trains (accuracy, loss or time) trains the returned copy in place, so a
        search can continue training from it. If only parameter count is weighted, the copy is
        untrained.
        """
        trained = copy.deepcopy(model)
        trained_in_use = False
        total = 0.0
        board = running_config.experiment_board
        board_metrics = board.simulation_metrics if board is not None else None
        if board_metrics is not None:
            board_metrics.clear()
            board_metrics["accuracy_metric"] = self.accuracy_metric.value
        for key, fn in self._SCORE_FUNCTIONS.items():
            weight = self.weights[key]
            if weight <= 0.0:
                continue
            if not trained_in_use and key in self._TRAINING_TERMS:
                term_model, trained_in_use = trained, True
            else:
                term_model = copy.deepcopy(model)
            term_score = fn(term_model, running_config)
            if board_metrics is not None:
                board_metrics[f"{key}_score"] = term_score
                board_metrics[f"{key}_weight"] = weight
                board_metrics[f"{key}_weighted"] = weight * term_score
            total += weight * term_score
        divisor = self.weight_sum()
        if math.isclose(divisor, 0.0):
            logger.error("SimulationScore.score: all weights are 0, returning 0 instead of NaN")
            return 0.0, trained
        composite = total / divisor
        if board_metrics is not None:
            board_metrics["composite_score"] = composite
        return composite, trained
