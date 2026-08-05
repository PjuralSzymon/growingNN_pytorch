"""Integration: grow very-small MNIST stem toward medium and big via concrete actions.

Architectures are copied here on purpose — tests must not import the experiments package.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.utils.layer_Factory import ConvFactory, LinearFactory
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph

# Set True to write FX PDF graphs under testResults/integration/...
SAVE_FX_GRAPHS = False

_CHANNELS = 4
_MNIST_SPATIAL = 28
_MAX_POOL_KERNEL = 2
_SPATIAL_AFTER_ONE_POOL = _MNIST_SPATIAL // _MAX_POOL_KERNEL  # 14
_SPATIAL_AFTER_TWO_POOLS = _SPATIAL_AFTER_ONE_POOL // _MAX_POOL_KERNEL  # 7
_FEATURES_ONE_POOL = _CHANNELS * _SPATIAL_AFTER_ONE_POOL ** 2  # 784
_FEATURES_TWO_POOLS = _CHANNELS * _SPATIAL_AFTER_TWO_POOLS ** 2  # 196
_GRAPH_DIR = "testResults/integration/architecture_change_very_small_to_medium_and_big"


class VerySmallMnistNet(nn.Module):
    """Big stem with conv2, second pool, and hidden linear removed."""

    def __init__(self, num_classes: int = 10, channels: int = _CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear2 = nn.Linear(_FEATURES_ONE_POOL, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, _MAX_POOL_KERNEL)
        return self.linear2(x.flatten(1))


class MediumMnistNet(nn.Module):
    """Big stem with conv2 and second pool removed (hidden linear kept)."""

    def __init__(self, num_classes: int = 10, channels: int = _CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(_FEATURES_ONE_POOL, _FEATURES_ONE_POOL, bias=True)
        self.linear2 = nn.Linear(_FEATURES_ONE_POOL, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, _MAX_POOL_KERNEL)
        x = self.linear(x.flatten(1))
        return self.linear2(x)


class BigMnistNet(nn.Module):
    """Target: two stem convs after pool + two hidden linears + classifier (one max-pool)."""

    def __init__(self, num_classes: int = 10, channels: int = _CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.linear = nn.Linear(_FEATURES_ONE_POOL, _FEATURES_ONE_POOL, bias=True)
        self.linear3 = nn.Linear(_FEATURES_ONE_POOL, _FEATURES_ONE_POOL, bias=True)
        self.linear2 = nn.Linear(_FEATURES_ONE_POOL, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, _MAX_POOL_KERNEL)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear(x.flatten(1))
        x = self.linear3(x)
        return self.linear2(x)


def architectures_equal(gm_a, gm_b) -> bool:
    """True when FX graphs match by op sequence (modules compared by type, not name)."""
    def sig(gm):
        return [
            ("module", type(gm.get_submodule(str(n.target))).__name__)
            if n.op == "call_module"
            else (n.op, getattr(n.target, "__name__", n.target))
            for n in gm.graph.nodes
        ]
    return sig(gm_a) == sig(gm_b)


def test_architecture_change_very_small_mnist_net_to_medium_and_big():
    """
    Action1: seq linear → medium.
    Actions 2–4: before-flatten seq conv, second seq conv, second seq linear → big.
    """
    # Arrange
    if SAVE_FX_GRAPHS:
        os.makedirs(_GRAPH_DIR, exist_ok=True)
    input_shape = (1, 1, _MNIST_SPATIAL, _MNIST_SPATIAL)
    very_small = TracedModel.create(VerySmallMnistNet(), input_shape)
    small_changed_to_medium = TracedModel.create(VerySmallMnistNet(), input_shape)
    small_changed_to_big = TracedModel.create(VerySmallMnistNet(), input_shape)
    medium = TracedModel.create(MediumMnistNet(), input_shape)
    big = TracedModel.create(BigMnistNet(), input_shape)

    # Act
    # Action 1 — small → medium: insert hidden Linear on conv1 → linear2
    action1 = AddSeqLinearLayer([
        "conv1",
        "linear2",
        LinearFactory.create_linear(_FEATURES_ONE_POOL, _FEATURES_ONE_POOL, Layer_Type.EYE),
        "seq_linear",
    ])
    action1.execute(small_changed_to_medium)
    action1.execute(small_changed_to_big)

    # Action 2 — small → big: first stem Conv before flatten on conv1 → seq_linear
    action2 = AddSeqConvLayer([
        "conv1",
        "seq_linear",
        ConvFactory.create_eye_conv(_CHANNELS, _CHANNELS, 3, stride=1, padding=1),
        "seq_conv",
    ])
    action2.execute(small_changed_to_big)

    # Action 3 — second stem Conv before flatten (conv3 role)
    action3 = AddSeqConvLayer([
        "conv1",
        "seq_linear",
        ConvFactory.create_eye_conv(_CHANNELS, _CHANNELS, 3, stride=1, padding=1),
        "seq_conv_2",
    ])
    action3.execute(small_changed_to_big)

    # Action 4 — second hidden Linear on seq_linear → linear2 (linear3 role)
    action4 = AddSeqLinearLayer([
        "seq_linear",
        "linear2",
        LinearFactory.create_linear(_FEATURES_ONE_POOL, _FEATURES_ONE_POOL, Layer_Type.EYE),
        "seq_linear_2",
    ])
    action4.execute(small_changed_to_big)

    # Assert
    if SAVE_FX_GRAPHS:
        draw_filtered_fx_graph(very_small.gm, f"{_GRAPH_DIR}/01_very_small", fmt="pdf")
        draw_filtered_fx_graph(small_changed_to_medium.gm, f"{_GRAPH_DIR}/02_small_changed_to_medium", fmt="pdf")
        draw_filtered_fx_graph(medium.gm, f"{_GRAPH_DIR}/03_medium", fmt="pdf")
        draw_filtered_fx_graph(small_changed_to_big.gm, f"{_GRAPH_DIR}/04_small_changed_to_big", fmt="pdf")
        draw_filtered_fx_graph(big.gm, f"{_GRAPH_DIR}/05_big", fmt="pdf")
    assert architectures_equal(small_changed_to_medium.gm, medium.gm)
    assert architectures_equal(small_changed_to_big.gm, big.gm)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
