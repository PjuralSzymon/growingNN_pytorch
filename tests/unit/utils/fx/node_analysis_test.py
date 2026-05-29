"""Unit tests for ``growingnn.utils.fx.node_analysis``."""

import operator
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.fx import ModuleResolver, NodeTypeChecker, NodeWidthAnalyser
from tests.model_factory import ModelFactory


def test_find_call_module_raises_for_missing_target():
    """
    find_call_module should raise ValueError when no call_module matches the name.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    nodes = list(gm.graph.nodes)

    # Act / Assert
    assert ModuleResolver.find_call_module(nodes, "l1") is not None
    with pytest.raises(ValueError, match="No call_module node"):
        ModuleResolver.find_call_module(nodes, "res1")


def test_get_layer_module_resolves_dotted_submodule_path():
    """
    get_layer_module should use get_submodule for nested targets like block.inner.
    """
    # Arrange
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(4, 4)

        def forward(self, x):
            return self.inner(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, x):
            return self.block(x)

    gm = fx.symbolic_trace(Model())

    # Act
    mod = ModuleResolver.get_layer_module("block.inner", gm)

    # Assert
    assert isinstance(mod, nn.Linear)


def test_get_layer_module_accepts_fx_node_target():
    """
    get_layer_module should accept an fx.Node and read its .target string.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    l1 = next(n for n in gm.graph.nodes if n.target == "l1")

    # Act
    mod = ModuleResolver.get_layer_module(l1, gm)

    # Assert
    assert isinstance(mod, nn.Linear)


def test_get_layer_module_returns_none_for_missing_path():
    """
    get_layer_module should return None when the submodule path does not exist.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act
    mod = ModuleResolver.get_layer_module("missing.layer", gm)

    # Assert
    assert mod is None


def test_unique_call_module_name_avoids_existing_targets():
    """
    unique_call_module_name should append a numeric suffix when base is already taken.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    gm.add_module("seq_linear_0", nn.Linear(4, 4))

    # Act
    name = ModuleResolver.unique_call_module_name("seq_linear", gm)

    # Assert
    assert name == "seq_linear_1"


def test_is_passthrough_true_for_relu_call_module():
    """
    is_passthrough should recognize nn.ReLU call_module nodes from PASSTHROUGH_MODULES.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3_with_activation())
    act_node = next(
        n for n in gm.graph.nodes
        if n.op == "call_module" and n.target == "act"
    )

    # Act / Assert
    assert NodeTypeChecker.is_passthrough(gm, act_node) is True


def test_is_fork_true_when_node_has_multiple_users():
    """
    is_fork should be true when a node feeds more than one successor.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())
    l1 = next(n for n in gm.graph.nodes if n.target == "l1")

    # Act / Assert
    assert NodeTypeChecker.is_fork(l1) is True


def test_is_add_true_for_operator_add_node():
    """
    is_add should detect operator.add call_function nodes.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())
    add_node = next(
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target == operator.add
    )

    # Act / Assert
    assert NodeTypeChecker.is_add(add_node) is True


def test_node_output_width_reads_linear_out_features():
    """
    node_output_width should return out_features for a Linear call_module node.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    l1 = next(n for n in gm.graph.nodes if n.target == "l1")

    # Act
    width = NodeWidthAnalyser.node_output_width(gm, l1)

    # Assert
    assert width == 4


def test_inputs_match_width_checks_all_inputs():
    """
    inputs_match_width should be true only when every input shares the same width.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())
    add_node = next(
        n for n in gm.graph.nodes
        if n.op == "call_function" and n.target == operator.add
    )

    # Act
    match = NodeWidthAnalyser.inputs_match_width(gm, add_node, 4)

    # Assert
    assert match is True


def test_all_sites_match_width_for_single_call_site():
    """
    all_sites_match_width should be true when the only call site inputs match width.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act / Assert
    assert NodeWidthAnalyser.all_sites_match_width(gm, "l2", 4) is True


def test_propagation_hits_unsizable_true_when_add_has_conv_sibling():
    """
    propagation_hits_unsizable should be true when forward path reaches add with Conv2d sibling.
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
    node = ModuleResolver.find_call_module(list(gm.graph.nodes), "linear_hidden")

    # Act
    hits = NodeWidthAnalyser.propagation_hits_unsizable(gm, node)

    # Assert
    assert hits is True


def test_propagation_hits_unsizable_false_for_plain_linear_chain():
    """
    propagation_hits_unsizable should be false on a simple linear chain middle layer.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    l2 = ModuleResolver.find_call_module(list(gm.graph.nodes), "l2")

    # Act / Assert
    assert NodeWidthAnalyser.propagation_hits_unsizable(gm, l2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
