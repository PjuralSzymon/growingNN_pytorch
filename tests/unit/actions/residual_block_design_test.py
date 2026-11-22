"""Checks for the residual-block action design: placement, quasi-identity copy, side inputs, shrink rules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.registry import generate_all_actions
from growingnn.actions.residual_block import ResidualBlockAction, discover_residual_regions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel


class _OneBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.block_conv = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.block_bn = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        x = F.relu(x + self.block_bn(self.block_conv(x)))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class _TwoBlocks(nn.Module):
    """Two blocks; identical when both kernels are 3, different when the second kernel is 1."""

    def __init__(self, second_kernel: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.c1 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, second_kernel, padding=second_kernel // 2, bias=False)
        self.b2 = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        x = F.relu(x + self.b1(self.c1(x)))
        x = F.relu(x + self.b2(self.c2(x)))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def _trained_like(model: nn.Module) -> TracedModel:
    """Give batch norms non-trivial affine terms and running statistics, then trace."""
    torch.manual_seed(0)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.3, 0.3)
    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(16, 3, 8, 8))
    return TracedModel.create(model, (1, 3, 8, 8))


def _add_side_input(traced: TracedModel, target: str) -> None:
    """Wire a zero residual conv from the stem into ``target`` (a side input into that block)."""
    action = next(a for a in AddResConvLayer.generate_all_actions(traced) if a.params[:2] == ["stem", target])
    action.execute(traced)
    traced.invalidate()


def _duplicate(traced: TracedModel) -> None:
    next(a for a in ResidualBlockAction.generate_all_actions(traced, shrink=False)).execute(traced)
    traced.invalidate()


def _operations(actions) -> list[str]:
    return [a.params[0] for a in actions]


def test_single_block_offers_duplicate_but_never_removes_the_last_block():
    """
    A network with one residual block should get a duplicate candidate and no shrink candidate.
    """

    # Arrange
    traced = _trained_like(_OneBlock())

    # Act
    actions = ResidualBlockAction.generate_all_actions(traced)

    # Assert
    assert _operations(actions) == ["duplicate"]


def test_duplicate_is_inserted_after_the_block_relu_with_its_own_relu():
    """
    The copy should read the activated block output and end with its own ReLU, so the network has three ReLUs.
    """

    # Arrange
    traced = _trained_like(_OneBlock())

    # Act
    _duplicate(traced)

    # Assert
    nodes = {node.name: node for node in traced.gm.graph.nodes}
    copied_conv = next(node for node in traced.gm.graph.nodes if str(node.target).startswith("block_conv_block"))
    assert copied_conv.args[0] is nodes["relu_1"]
    assert sum(1 for n in traced.gm.graph.nodes if n.target is F.relu) == 3


def test_duplicate_uses_quasi_identity_weights():
    """
    The copied conv should be a centred identity kernel and the copied batch norm should be reset.
    """

    # Arrange
    traced = _trained_like(_OneBlock())

    # Act
    _duplicate(traced)

    # Assert
    conv = next(m for name, m in traced.gm.named_modules() if name.startswith("block_conv_block"))
    norm = next(m for name, m in traced.gm.named_modules() if name.startswith("block_bn_block"))
    expected = torch.zeros_like(conv.weight)
    for channel in range(8):
        expected[channel, channel, 1, 1] = 1.0
    assert torch.equal(conv.weight, expected)
    assert torch.equal(norm.weight, torch.ones(8)) and torch.equal(norm.bias, torch.zeros(8))
    assert torch.equal(norm.running_mean, torch.zeros(8)) and torch.equal(norm.running_var, torch.ones(8))


def test_block_with_side_input_from_add_res_conv_is_still_discovered():
    """
    After AddResConvLayer wires the stem into the block, the block should still be a valid region.
    """

    # Arrange
    traced = _trained_like(_OneBlock())
    _add_side_input(traced, "block_conv")

    # Act
    regions = [r for r in discover_residual_regions(traced) if r[3] is None]

    # Assert
    assert len(regions) == 1


def test_duplicate_of_block_with_side_input_does_not_copy_the_side_input():
    """
    The copy should compute only y + F(y): the side conv keeps one user and is not duplicated.
    """

    # Arrange
    traced = _trained_like(_OneBlock())
    _add_side_input(traced, "block_conv")
    probe = torch.randn(4, 3, 8, 8)
    expected_shape = traced.gm(probe).shape

    # Act
    _duplicate(traced)

    # Assert
    side_nodes = [n for n in traced.gm.graph.nodes if n.op == "call_module" and "res_conv" in str(n.target)]
    assert len(side_nodes) == 1
    assert len(side_nodes[0].users) == 1
    traced.gm.eval()
    assert traced.gm(probe).shape == expected_shape


def test_two_different_blocks_offer_remove_for_each_block():
    """
    Two blocks with different shapes cannot be merged, so each should get a remove candidate.
    """

    # Arrange
    traced = _trained_like(_TwoBlocks(second_kernel=1))

    # Act
    actions = ResidualBlockAction.generate_all_actions(traced, grow=False)

    # Assert
    assert _operations(actions) == ["remove", "remove"]


def test_identical_adjacent_blocks_offer_merge_instead_of_remove():
    """
    Two adjacent identical blocks should get one merge candidate and no remove candidates.
    """

    # Arrange
    traced = _trained_like(_TwoBlocks())

    # Act
    actions = ResidualBlockAction.generate_all_actions(traced, grow=False)

    # Assert
    assert _operations(actions) == ["merge"]
    assert len(actions[0].params) == 5


def test_merge_folds_second_block_into_first():
    """
    Merging should leave one block whose last norm holds the summed affine terms of both blocks.
    """

    # Arrange
    traced = _trained_like(_TwoBlocks())
    gamma_sum = (traced.gm.b1.weight + traced.gm.b2.weight).detach().clone()
    params_before = sum(p.numel() for p in traced.gm.parameters())
    probe = torch.randn(4, 3, 8, 8)

    # Act
    ResidualBlockAction.generate_all_actions(traced, grow=False)[0].execute(traced)

    # Assert
    targets = {str(node.target) for node in traced.gm.graph.nodes if node.op == "call_module"}
    assert "c2" not in targets and "b2" not in targets
    assert torch.allclose(traced.gm.b1.weight, gamma_sum)
    assert sum(p.numel() for p in traced.gm.parameters()) < params_before
    traced.gm.eval()
    assert traced.gm(probe).shape == (4, 5)


def test_duplicate_then_shrink_merges_back_to_one_block():
    """
    After a duplicate, the shrink candidate should be a merge that returns the network to one block.
    """

    # Arrange
    traced = _trained_like(_OneBlock())
    _duplicate(traced)

    # Act
    shrinks = ResidualBlockAction.generate_all_actions(traced, grow=False)
    shrinks[0].execute(traced)
    traced.invalidate()

    # Assert
    assert _operations(shrinks) == ["merge"]
    assert len([r for r in discover_residual_regions(traced) if r[3] is None]) == 1


def test_remove_deletes_the_side_branch_of_the_removed_block():
    """
    Removing a block with a side input should also delete the side conv that fed only that block.
    """

    # Arrange
    traced = _trained_like(_TwoBlocks(second_kernel=1))
    _add_side_input(traced, "c1")
    probe = torch.randn(4, 3, 8, 8)
    expected_shape = traced.gm(probe).shape

    # Act
    remove_first = next(
        a for a in ResidualBlockAction.generate_all_actions(traced, grow=False) if a.params[0] == "remove"
    )
    remove_first.execute(traced)

    # Assert
    assert not any("res_conv" in str(node.target) for node in traced.gm.graph.nodes)
    traced.gm.eval()
    assert traced.gm(probe).shape == expected_shape


def test_registry_respects_residual_block_grow_and_shrink_flags():
    """
    The registry should offer block shrink candidates only when residual-block shrink is enabled.
    """

    # Arrange
    traced = _trained_like(_TwoBlocks())
    config = RunningConfig(generations=1, epochs=1)

    # Act
    with_shrink = [a for a in generate_all_actions(traced, config) if isinstance(a, ResidualBlockAction)]
    config.ACTIONS_ENABLE_RESIDUAL_BLOCK_SHRINK = False
    without_shrink = [a for a in generate_all_actions(traced, config) if isinstance(a, ResidualBlockAction)]

    # Assert
    assert _operations(with_shrink) == ["duplicate", "duplicate", "merge"]
    assert _operations(without_shrink) == ["duplicate", "duplicate"]
