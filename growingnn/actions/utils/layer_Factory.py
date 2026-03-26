"""Factory for building layers from :class:`Layer_Type` (stubs; fill in init logic later)."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch.nn as nn


class Layer_Type(Enum):
    ZERO = 1
    RANDOM = 2
    EYE = 3

class LinearFactory:

    @staticmethod
    def create_zero_linear(in_features: int, out_features: int) -> nn.Linear:
        layer = nn.Linear(in_features, out_features)
        layer.weight.data.zero_()
        layer.bias.data.zero_()
        return layer

    @staticmethod
    def create_random_linear(in_features: int, out_features: int) -> nn.Linear:
        layer = nn.Linear(in_features, out_features)
        layer.weight.data.random_(0, 1)
        layer.bias.data.random_(0, 1)
        return layer

    @staticmethod
    def create_eye_linear(in_features: int, out_features: int) -> nn.Linear:
        return nn.Linear(self.in_features, self.out_features)

