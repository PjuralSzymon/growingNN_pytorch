"""Factory for building layers from :class:`Layer_Type` (stubs; fill in init logic later)."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
import torch.nn as nn

from growingnn.actions.utils import quaziIdentity


class Layer_Type(Enum):
    ZERO = 1
    RANDOM = 2
    EYE = 3

class LinearFactory:

    @staticmethod
    def create_linear(in_features: int, out_features: int, type: Layer_Type) -> nn.Linear:
        if type == Layer_Type.ZERO:
            return LinearFactory.create_zero_linear(in_features, out_features)
        elif type == Layer_Type.RANDOM:
            return LinearFactory.create_random_linear(in_features, out_features)
        elif type == Layer_Type.EYE:
            return LinearFactory.create_eye_linear(in_features, out_features)
        else:
            raise ValueError(f"Unsupported layer type: {type}")

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
        layer = nn.Linear(in_features, out_features)
        w = quaziIdentity.eye_stretch(in_features, out_features)
        # Linear.weight is (out_features, in_features); eye_stretch returns (in_features, out_features)
        layer.weight.data = torch.as_tensor(w, dtype=layer.weight.dtype).T.contiguous()
        layer.bias.data.zero_()
        return layer

