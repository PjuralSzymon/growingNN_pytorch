"""Shape-preserving low-rank feature adapters for FX graph actions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LowRankFeatureAdapter(nn.Module):
    """Apply a trainable low-rank update along a selected feature dimension.

    With ``residual=True``, the initial module is exactly the identity. With
    ``residual=False``, its initial output is zero, which is suitable for a new
    residual branch. All other dimensions (batch, time, spatial, etc.) pass through.
    """

    def __init__(
        self,
        features: int,
        rank: int,
        *,
        residual: bool,
        feature_axis: int = -1,
    ):
        super().__init__()
        if features <= 0 or rank <= 0 or rank > features:
            raise ValueError("features and rank must be positive, with rank <= features")
        self.down_weight = nn.Parameter(torch.empty(rank, features))
        self.up_weight = nn.Parameter(torch.zeros(features, rank))
        nn.init.kaiming_uniform_(self.down_weight, a=5**0.5)
        self.residual = residual
        self.feature_axis = feature_axis

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        axis = self.feature_axis if self.feature_axis >= 0 else value.ndim + self.feature_axis
        if axis < 0 or axis >= value.ndim:
            raise ValueError(
                f"feature axis {self.feature_axis} is invalid for rank-{value.ndim} input"
            )
        moved = value.movedim(axis, -1)
        update = F.linear(F.linear(moved, self.down_weight), self.up_weight)
        output = moved + update if self.residual else update
        return output.movedim(-1, axis)


class FeatureAxisConv2dAdapter(nn.Module):
    """Shape-preserving 1x1 channel adapter for a rank-4 tensor boundary.

    ``feature_axis`` is explicit because rank four alone does not distinguish
    NCHW, NHWC, or other task-specific layouts. Sequential instances start as
    identity; residual-branch instances start at zero.
    """

    def __init__(
        self,
        features: int,
        *,
        feature_axis: int,
        residual: bool,
    ):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.feature_axis = feature_axis
        self.conv = nn.Conv2d(features, features, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            if residual:
                diagonal = torch.arange(features)
                self.conv.weight[diagonal, diagonal, 0, 0] = 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(
                f"FeatureAxisConv2dAdapter requires rank 4, got rank {value.ndim}"
            )
        axis = self.feature_axis if self.feature_axis >= 0 else value.ndim + self.feature_axis
        if axis < 0 or axis >= value.ndim or axis == 0:
            raise ValueError(
                f"feature axis {self.feature_axis} is invalid for rank-4 input"
            )
        moved = value.movedim(axis, 1)
        return self.conv(moved).movedim(1, axis)


def supported_adapter_ranks(features: int, max_parameters: int) -> tuple[int, ...]:
    """Small useful ranks that respect the configured per-action parameter cap."""
    return tuple(
        rank for rank in (4, 8, 16)
        if rank <= features and 2 * features * rank <= max_parameters
    )
