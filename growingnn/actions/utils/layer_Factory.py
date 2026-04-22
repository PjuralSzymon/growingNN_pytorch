"""Factory for building layers from :class:`Layer_Type` (stubs; fill in init logic later)."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from growingnn import config
from growingnn.actions.utils import quaziIdentity
from growingnn.config import ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE


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
        mean, std = ADDING_RES_LAYERS_WEIGHT_INITIALIZATION_RANGE
        layer = nn.Linear(in_features, out_features)
        layer.weight.data.normal_(mean, std)
        layer.bias.data.normal_(mean, std)
        return layer

    @staticmethod
    def create_eye_linear(in_features: int, out_features: int) -> nn.Linear:
        layer = nn.Linear(in_features, out_features)
        w = quaziIdentity.eye_stretch(in_features, out_features)
        # Linear.weight is (out_features, in_features); eye_stretch returns (in_features, out_features)
        layer.weight.data = torch.as_tensor(w, dtype=layer.weight.dtype).T.contiguous()
        layer.bias.data.zero_()
        return layer

class ConvFactory:

    @staticmethod
    def create_conv(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, type: Layer_Type) -> nn.Conv2d:
        if type == Layer_Type.ZERO:
            return ConvFactory.create_zero_conv(in_channels, out_channels, kernel_size, stride, padding)
        if type == Layer_Type.EYE:
            return ConvFactory.create_eye_conv(in_channels, out_channels, kernel_size, stride, padding)
        else:
            raise ValueError(f"Unsupported layer type: {type}")

    @staticmethod
    def create_zero_conv(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Conv2d:
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        layer.weight.data.zero_()
        layer.bias.data.zero_()
        return layer

    @staticmethod
    def create_eye_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
    ) -> nn.Conv2d:
        # Spatial delta kernel (1 at centre, 0 elsewhere) on the in/out diagonal; not a dense channel mix.
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        layer.weight.data.zero_()
        kh, kw = _pair(kernel_size)
        ch, cw = (kh - 1) // 2, (kw - 1) // 2
        for i in range(min(in_channels, out_channels)):
            layer.weight.data[i, i, ch, cw] = 1.0
        layer.bias.data.zero_()
        return layer

    @staticmethod
    def create_zero_conv_before_linear(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Conv2d:
        layer = ConvFactory.create_zero_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if config.RES_CONV_TO_LINEAR_GLOBAL_POOL_TYPE == "max":
            pool = nn.AdaptiveMaxPool2d(1)
        else:
            pool = nn.AdaptiveAvgPool2d(1)
        layer = nn.Sequential(
            layer,
            pool,
            nn.Flatten(start_dim=1),
        )
        return layer