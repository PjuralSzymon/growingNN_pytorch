"""Action generation for a simple MNIST-style CNN."""

import sys
from pathlib import Path

import pytest
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from tests.model_factory import ModelFactory


def test_simple_mnist_cnn_generates_conv_grow_actions():
    """
    For this CNN the simulator should propose convolution-based grow actions,
    not only Add Seq Linear Layer.

    Architecture (input shape N,1,28,28):
      conv1  Conv2d(1->3, k=3)
        -> relu -> max_pool2d
      conv2  Conv2d(3->3, k=3)
        -> relu -> max_pool2d
      adaptive_avg_pool2d -> flatten
      linear Linear(3->10)
        -> output (N,10)

    Expected candidates include Add Seq Conv Layer and Add Res Conv Layer.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_mnist_cnn())
    config = RunningConfig(generations=1, epochs=1)

    # Act
    actions = generate_all_actions(gm, config)
    action_types = [type(action).__name__ for action in actions]
    conv_actions = [
        action
        for action in actions
        if isinstance(action, (AddSeqConvLayer, AddResConvLayer))
    ]

    # Assert
    assert conv_actions, (
        "Expected conv grow actions for the simple MNIST CNN, "
        f"but generate_all_actions returned only: {action_types}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

