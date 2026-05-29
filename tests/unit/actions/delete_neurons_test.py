"""Unit tests for ``growingnn.actions.delete_neurons`` neuron-shrink propagation."""
import sys
from pathlib import Path

import torch
import torch.fx as fx
import pytest
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.delete_neurons import DelNeurons, resize_layer_output, shrink_layer_output
from tests.model_factory import ModelFactory


def test_resize_layer_output_raises_for_non_linear_module():
    """
    resize_layer_output should raise TypeError when the target is not nn.Linear.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())

    # Act / Assert
    with pytest.raises(TypeError, match="not nn.Linear"):
        resize_layer_output(gm, "c1", 2)


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


def test_shrink_propagates_through_long_add_chain_to_downstream_linear():
    """
    Shrinking a linear whose output passes through 10+ chained add nodes
    (each with an independent linear sibling) must update the downstream
    linear's in_features at the end of the chain.
    """

    # Arrange
    N_ADDS = 12

    class LongAddChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.start = nn.Linear(10, 20)
            self.end = nn.Linear(20, 5)
            for i in range(N_ADDS):
                setattr(self, f"side_{i}", nn.Linear(10, 20))

        def forward(self, x):
            out = self.start(x)
            for i in range(N_ADDS):
                out = out + getattr(self, f"side_{i}")(x)
            return self.end(out)

    gm = fx.symbolic_trace(LongAddChain())
    x = torch.randn(2, 10)

    # Act
    shrink_layer_output(gm, "start", 0.5)
    y = gm(x)

    # Assert
    assert gm.start.out_features == 10
    assert gm.end.in_features == 10
    for i in range(N_ADDS):
        assert getattr(gm, f"side_{i}").out_features == 10
    assert y.shape == (2, 5)


def test_shrink_propagates_through_add_chain_with_forked_siblings():
    """
    Same long-add-chain pattern but sibling layers draw from a shared fork
    source (like res_conv layers fed by a shared conv). The fork source must
    NOT be resized, but each sibling's output must be shrunk.
    """

    # Arrange
    N_ADDS = 8

    class ForkSiblingChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.start = nn.Linear(10, 20)
            self.source = nn.Linear(10, 20)
            self.end = nn.Linear(20, 5)
            for i in range(N_ADDS):
                setattr(self, f"side_{i}", nn.Linear(20, 20))

        def forward(self, x):
            out = self.start(x)
            s = self.source(x)
            for i in range(N_ADDS):
                out = out + getattr(self, f"side_{i}")(s)
            return self.end(out)

    gm = fx.symbolic_trace(ForkSiblingChain())
    x = torch.randn(2, 10)

    # Act
    shrink_layer_output(gm, "start", 0.5)
    y = gm(x)

    # Assert
    assert gm.start.out_features == 10
    assert gm.end.in_features == 10
    assert gm.source.out_features == 20, "fork source must stay unchanged"
    for i in range(N_ADDS):
        side = getattr(gm, f"side_{i}")
        assert side.out_features == 10
        assert side.in_features == 20, "input from fork source stays unchanged"
    assert y.shape == (2, 5)


def test_generate_actions_skips_linear_with_conv_sibling_at_add():
    """
    DelNeurons.generate_all_actions must NOT produce an action for a Linear
    whose forward propagation reaches an add node where the sibling branch
    is a Conv2d (non-sizable). This prevents runtime shape mismatches.
    """

    # Arrange
    class LinearConvAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 10, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear_hidden = nn.Linear(10, 10)
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            conv_out = self.pool(self.conv(x)).flatten(1)
            lin_out = self.linear_hidden(conv_out)
            return self.fc(lin_out + conv_out)

    gm = fx.symbolic_trace(LinearConvAdd())

    # Act
    actions = DelNeurons.generate_all_actions(gm)

    # Assert
    layer_ids = [a.params[0] for a in actions]
    print("test_generate_actions_skips_linear_with_conv_sibling_at_add: layer_ids: %s", layer_ids)
    assert "linear_hidden" not in layer_ids, (
        "linear_hidden feeds an add with a conv sibling — must be filtered out"
    )


def test_generate_actions_keeps_linear_without_conv_sibling():
    """
    DelNeurons.generate_all_actions must still produce actions for Linears
    whose propagation path only touches other sizable (Linear/BN) siblings.
    """

    # Arrange — stem makes l1/l2 hidden (not directly fed by placeholder)
    class PureLinearAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(10, 20)
            self.l1 = nn.Linear(20, 20)
            self.l2 = nn.Linear(20, 20)
            self.fc = nn.Linear(20, 5)

        def forward(self, x):
            h = self.stem(x)
            return self.fc(self.l1(h) + self.l2(h))

    gm = fx.symbolic_trace(PureLinearAdd())

    # Act
    actions = DelNeurons.generate_all_actions(gm)

    # Assert
    layer_ids = [a.params[0] for a in actions]
    assert "l1" in layer_ids
    assert "l2" in layer_ids


def test_generate_actions_allows_linear_after_conv_when_add_sibling_is_linear():
    """
    conv -> pool -> flatten -> linear_a -> add -> fc
                                  linear_b --/

    The add sibling (linear_b) is sizable so linear_a is safe to shrink.
    Conv is upstream of linear_a (not a sibling at the add).
    """

    # Arrange
    class ConvThenLinearAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.linear_a = nn.Linear(16, 20)
            self.linear_b = nn.Linear(16, 20)
            self.fc = nn.Linear(20, 5)

        def forward(self, x):
            h = self.pool(self.conv(x)).flatten(1)
            return self.fc(self.linear_a(h) + self.linear_b(h))

    gm = fx.symbolic_trace(ConvThenLinearAdd())

    # Act
    actions = DelNeurons.generate_all_actions(gm)

    # Assert
    layer_ids = [a.params[0] for a in actions]
    assert "linear_a" in layer_ids
    assert "linear_b" in layer_ids


def test_generate_actions_allows_nested_linear_chain_after_conv():
    """
    Deeply nested case: conv backbone feeds into a long chain of
    linear layers with multiple add nodes, all siblings are linear.
    All hidden linears should be shrinkable.

    conv -> pool -> flatten -> stem -> l1 -> add -> l3 -> add -> l5 -> fc
                                       l2 --/             l4 --/
    """

    # Arrange
    class NestedLinearAfterConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.stem = nn.Linear(16, 32)
            self.l1 = nn.Linear(32, 32)
            self.l2 = nn.Linear(32, 32)
            self.l3 = nn.Linear(32, 32)
            self.l4 = nn.Linear(32, 32)
            self.l5 = nn.Linear(32, 32)
            self.fc = nn.Linear(32, 5)

        def forward(self, x):
            h = self.pool(self.conv(x)).flatten(1)
            h = self.stem(h)
            h = self.l1(h) + self.l2(h)
            h = self.l3(h) + self.l4(h)
            return self.fc(self.l5(h))

    gm = fx.symbolic_trace(NestedLinearAfterConv())
    x = torch.randn(2, 3, 8, 8)

    # Act
    actions = DelNeurons.generate_all_actions(gm)
    layer_ids = [a.params[0] for a in actions]

    # Assert — all hidden linears are safe (conv is upstream, not a sibling)
    for name in ("l1", "l2", "l3", "l4", "l5"):
        assert name in layer_ids, f"{name} should be shrinkable"

    # Verify shrinking actually works end-to-end
    shrink_layer_output(gm, "l1", 0.5)
    y = gm(x)
    assert gm.l1.out_features == 16
    assert gm.l2.out_features == 16
    assert gm.l3.in_features == 16
    assert y.shape == (2, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
