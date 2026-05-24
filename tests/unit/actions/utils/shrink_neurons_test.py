"""Unit tests for neuron-shrink shape propagation."""
import sys
from pathlib import Path

import torch
import torch.fx as fx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.delete_neurons import DelNeurons
from growingnn.actions.utils.shrink_neurons import shrink_layer_output
from tests.model_factory import ModelFactory


def test_shrink_linear_chain_updates_downstream_input():
    """
    Shrinking a hidden linear output should update the next linear in_features.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    x = torch.randn(2, 4)

    # Act
    shrink_layer_output(gm, "l2", 0.5)
    y = gm(x)

    # Assert
    assert gm.l2.out_features == 2
    assert gm.l3.in_features == 2
    assert y.shape == (2, 4)


def test_shrink_residual_branch_updates_fork_linear_input():
    """
    Shrinking r4_b should align merge skip width and r4_a in_features at the fork.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths())
    x = torch.randn(2, 4)

    # Act
    DelNeurons(["r4_b", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.r4_b.out_features == 5
    assert gm.r4_a.in_features == gm.merge.out_features
    assert gm.head.in_features == gm.r4_b.out_features
    assert y.shape == (2, 4)


def test_shrink_residual_skip_keeps_add_inputs_aligned():
    """
    Shrinking one branch before add should keep both add inputs the same width.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())
    x = torch.randn(2, 4)

    # Act
    DelNeurons(["l2", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.l2.out_features == 2
    assert gm.l4.out_features == 2
    assert y.shape == (2, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
