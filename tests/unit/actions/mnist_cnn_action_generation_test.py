"""Action generation for a simple MNIST-style CNN."""

import sys
from pathlib import Path

import pytest
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from tests.model_factory import ModelFactory


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
    config = RunningConfig(generations=1, epochs=1)

    # Act
    actions = generate_all_actions(gm, config)
    print(f"Available actions ({len(actions)}):")
    for i, action in enumerate(actions, start=1):
        print(f"  {i}. {type(action).__name__}: {action}")
    seq_linear_actions = [
        action for action in actions if isinstance(action, AddSeqLinearLayer)
    ]

    # Assert
    assert len(seq_linear_actions) == 1
    action = seq_linear_actions[0]
    assert action.params[0] == "conv1"
    assert action.params[1] == "linear"
    assert isinstance(action.params[2], nn.Linear)
    assert action.params[2].in_features == 3
    assert action.params[2].out_features == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

