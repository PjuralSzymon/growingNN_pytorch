"""Factories for shape-preserving regularization modules."""

from __future__ import annotations

import torch.nn as nn


class RegularizationFactory:
    @staticmethod
    def create_dropout(shape: tuple[int, ...], p: float) -> nn.Module | None:
        """Return Dropout for rank-2 activations or Dropout2d for rank-4; None when unsupported."""
        if len(shape) == 2:
            return nn.Dropout(p)
        if len(shape) == 4:
            return nn.Dropout2d(p)
        return None
