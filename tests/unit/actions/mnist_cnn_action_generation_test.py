"""Action generation for a simple MNIST-style CNN."""

import sys
from pathlib import Path

import pytest
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from tests.model_factory import ModelFactory

try:
    from growingnn.core.traced_model import TracedModel
except ImportError:
    TracedModel = None


def _action_graph(gm: fx.GraphModule):
    """Return TracedModel on main; fall back to GraphModule on older branches."""
    if TracedModel is None:
        return gm
    return TracedModel.create(gm, (1, 1, 28, 28))


def test_simple_mnist_cnn_generates_seq_linear_between_boundary_conv_and_linear():
    """
    Sequential grow actions should work between input-boundary conv and output-boundary linear.

    Architecture (input shape N,1,28,28):
      conv1  Conv2d(1->3, k=3)
        -> relu -> max_pool2d
      adaptive_avg_pool2d -> flatten
      linear Linear(3->10)
        -> output (N,10)
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_mnist_cnn())
    graph = _action_graph(gm)

    # Act
    seq_linear_actions = AddSeqLinearLayer.generate_all_actions(graph)
    seq_conv_actions = AddSeqConvLayer.generate_all_actions(graph)
    res_conv_actions = AddResConvLayer.generate_all_actions(graph)

    # Assert
    assert len(seq_linear_actions) == 1
    action = seq_linear_actions[0]
    assert action.params[0] == "conv1"
    assert action.params[1] == "linear"
    assert isinstance(action.params[2], nn.Linear)
    assert action.params[2].in_features == 3
    assert action.params[2].out_features == 3
    assert len(seq_conv_actions) >= 1
    assert seq_conv_actions[0].params[0] == "conv1"
    assert seq_conv_actions[0].params[1] == "linear"
    assert isinstance(seq_conv_actions[0].params[2], nn.Conv2d)
    assert res_conv_actions == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
