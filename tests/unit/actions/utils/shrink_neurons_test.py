"""Unit tests for neuron-shrink shape propagation."""
import sys
from pathlib import Path

import torch
import torch.fx as fx
import pytest
import torch.nn as nn

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
    assert gm.l4.out_features == gm.l3.out_features
    assert y.shape == (2, 4)


def test_repeated_shrink_sequence_does_not_corrupt_stem_input():
    """
    A long random shrink sequence (regression-like) must keep stem.in_features at 4.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths())
    x = torch.randn(2, 4)
    sequence = [
        "r4_b",
        "r1_down",
        "r1_up",
        "r2_b",
        "r2_a",
        "r2_a",
        "expand",
        "r1_down",
        "r4_b",
        "merge",
        "r1_down",
        "r3_b",
        "r1_up",
        "r1_up",
        "expand",
    ]

    # Act
    for layer_id in sequence:
        DelNeurons([layer_id, 0.5]).execute(gm)
        gm(x)

    # Assert
    assert gm.stem.in_features == 4
    assert x.shape == gm(x).shape


def test_shrink_parallel_branch_does_not_shrink_fork_linear_input():
    """
    Shrinking one branch after a shared fork linear must keep fork.in_features unchanged.
    """

    # Arrange
    class ForkHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(512, 512)
            self.main = nn.Linear(512, 512)
            self.skip = nn.Linear(512, 512)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            h = self.shared(x)
            return self.fc(self.main(h) + self.skip(h))

    gm = fx.symbolic_trace(ForkHead())
    x = torch.randn(2, 512)

    # Act
    DelNeurons(["skip", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.shared.in_features == 512
    assert gm.skip.out_features == 256
    assert gm.main.out_features == 256
    assert y.shape == (2, 10)


def test_shrink_at_add_does_not_narrow_shared_fork_hub():
    """
    Shrinking one branch off a shared hub must not change hub.out or sibling branch inputs.
    """

    # Arrange
    class HubFork(nn.Module):
        def __init__(self):
            super().__init__()
            self.hub = nn.Linear(512, 512)
            self.a = nn.Linear(512, 256)
            self.b = nn.Linear(512, 256)
            self.tail = nn.Linear(256, 10)

        def forward(self, x):
            h = self.hub(x)
            bundle = self.a(h) + self.b(h)
            return self.tail(bundle)

    gm = fx.symbolic_trace(HubFork())
    x = torch.randn(2, 512)

    # Act
    DelNeurons(["a", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.hub.out_features == 512
    assert gm.a.out_features == 128
    assert gm.b.out_features == 128
    assert gm.tail.in_features == 128
    assert y.shape == (2, 10)


def test_branch_shrink_propagates_to_downstream_linear():
    """
    Shrinking a branch at an add must update later linears that consume that branch output.
    """

    # Arrange
    class ForkConsume(nn.Module):
        def __init__(self):
            super().__init__()
            self.s1 = nn.Linear(512, 512)
            self.s2 = nn.Linear(512, 512)
            self.mid = nn.Linear(512, 256)
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            bundle = self.s1(x) + self.s2(x)
            t = self.mid(bundle)
            return self.fc(t)

    gm = fx.symbolic_trace(ForkConsume())
    x = torch.randn(2, 512)

    # Act
    DelNeurons(["s1", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.s1.out_features == 256
    assert gm.s2.out_features == 256
    assert gm.mid.in_features == 256
    assert gm.fc.in_features == 256
    assert y.shape == (2, 10)


def test_shrink_syncs_nested_add_branches():
    """
    Shrinking one side of an outer add must fan into nested add inputs too.
    """

    # Arrange
    class NestedAddHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Linear(512, 512)
            self.branch_a = nn.Linear(512, 512)
            self.branch_b = nn.Linear(512, 512)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            inner = self.branch_a(x) + self.branch_b(x)
            return self.fc(self.main(x) + inner)

    gm = fx.symbolic_trace(NestedAddHead())
    x = torch.randn(2, 512)

    # Act
    DelNeurons(["main", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.main.out_features == 256
    assert gm.branch_a.out_features == 256
    assert gm.branch_b.out_features == 256
    assert y.shape == (2, 10)


def test_shrink_upstream_fork_updates_parallel_residual_branch():
    """
    Shrinking expand after r2_b should keep fork and branch widths equal at add_1.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths())
    x = torch.randn(2, 4)
    DelNeurons(["r2_b", 0.5]).execute(gm)

    # Act
    DelNeurons(["expand", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    assert gm.expand.out_features == gm.r2_b.out_features
    assert y.shape == (2, 4)


def test_shrink_nested_add_tree_syncs_fork_linear_at_distant_add():
    """
    ResNet-head pattern: hub fork, nested adds, and a linear fork (eye_9) feeding
    both a distant add and a side branch. Shrinking eye_7 must align add_17 inputs.
    """

    # Arrange
    class ResnetLikeNestedHead(nn.Module):
        """Miniature of the ResNet FX head around add_15..add_17 (see regression graph)."""

        def __init__(self):
            super().__init__()
            self.hub = nn.Linear(512, 512)
            self.side = nn.Linear(512, 512)
            self.eye_2 = nn.Linear(512, 128)
            self.eye_7 = nn.Linear(512, 128)
            self.eye_5 = nn.Linear(512, 128)
            self.eye_11 = nn.Linear(512, 128)
            self.seq_0 = nn.Linear(128, 128)
            self.eye_13 = nn.Linear(128, 128)
            self.eye_12 = nn.Linear(128, 128)
            self.eye_1 = nn.Linear(128, 128)
            self.eye_9 = nn.Linear(128, 128)
            self.eye_8 = nn.Linear(128, 128)
            self.eye_10 = nn.Linear(128, 128)
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            h = self.hub(x)
            s = self.side(x)
            add_19 = self.eye_5(h) + self.eye_11(h)
            add_15 = self.eye_2(s) + self.eye_7(h)
            add_21 = self.seq_0(add_19) + self.eye_13(add_19)
            add_20 = add_21 + self.eye_12(add_19)
            add_10 = add_20 + add_15
            eye_1 = self.eye_1(add_15)
            eye_9 = self.eye_9(add_19)
            add_17 = eye_1 + eye_9
            add_16 = add_17 + self.eye_8(add_19)
            add_9 = add_10 + add_16
            skip = self.eye_10(eye_9)
            return self.fc(add_9 + skip)

    gm = fx.symbolic_trace(ResnetLikeNestedHead())
    x = torch.randn(2, 512)

    # Act
    DelNeurons(["eye_7", 0.5]).execute(gm)
    y = gm(x)

    # Assert
    w = gm.eye_7.out_features
    assert gm.eye_1.out_features == w
    assert gm.eye_9.out_features == w
    assert gm.eye_10.in_features == w
    assert gm.hub.out_features == 512
    assert y.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
