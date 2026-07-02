"""Unit tests for ``growingnn.utils.fx.graph_editor``."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.utils.layer_Factory import Layer_Type, LinearFactory
from growingnn.utils.fx import ModelStructureEditor, ModuleResolver
from growingnn.utils.fx.sum_nodes import nary_add
from tests.model_factory import ModelFactory


def test_add_new_residual_layer_zero_branch_preserves_output():
    """
    A zero-weight residual branch should leave the forward pass unchanged.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    x = torch.randn(1, 4)
    y_initial = gm(x)
    layer = nn.Linear(4, 4)
    layer.weight.data.zero_()
    layer.bias.data.zero_()

    # Act
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", layer, name="res1")
    y_after = gm(x)

    # Assert
    assert torch.allclose(y_after, y_initial)


def test_add_new_residual_layer_adds_call_module_node():
    """
    add_new_residual_layer should insert a new call_module target into the graph.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", nn.Linear(4, 4), name="res1")
    nodes = list(gm.graph.nodes)

    # Assert
    assert ModuleResolver.find_call_module(nodes, "res1") is not None


def test_add_new_residual_layer_uses_single_nary_add_for_multiple_branches():
    """
    Multiple residual branches at the same dst should share one nary_add node.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    x = torch.randn(1, 4)
    y0 = gm(x)
    layer1 = nn.Linear(4, 4)
    layer1.weight.data.zero_()
    layer1.bias.data.zero_()
    layer2 = nn.Linear(4, 4)
    layer2.weight.data.zero_()
    layer2.bias.data.zero_()

    # Act
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", layer1, name="res1")
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", layer2, name="res2")
    nary_nodes = [node for node in gm.graph.nodes if node.op == "call_function" and node.target is nary_add]

    # Assert
    assert torch.allclose(gm(x), y0)
    assert len(nary_nodes) == 1
    assert len(nary_nodes[0].args) == 3


def test_add_new_residual_layer_flattens_existing_binary_add():
    """
    A residual at dst should flatten add(add(a, b), dst) into one nary_add with all branches.
    """
    # Arrange
    class NestedAddModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 4)
            self.contract = nn.Linear(4, 4)

        def forward(self, x):
            a = self.l1(x)
            b = self.l2(a)
            mid = a + b
            c = self.contract(mid)
            return mid + c

    gm = fx.symbolic_trace(NestedAddModel())
    x = torch.randn(2, 4)
    y0 = gm(x)
    layer = nn.Linear(4, 4)
    layer.weight.data.zero_()
    layer.bias.data.zero_()

    # Act
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "contract", layer, name="res1")
    nary_nodes = [node for node in gm.graph.nodes if node.op == "call_function" and node.target is nary_add]
    output_sum = next(arg for node in gm.graph.nodes if node.op == "output" for arg in node.args)

    # Assert
    assert torch.allclose(gm(x), y0)
    assert len(nary_nodes) == 1
    assert len(nary_nodes[0].args) == 4
    assert output_sum is nary_nodes[0]


def test_add_new_seq_layer_eye_preserves_output_simple_chain():
    """
    An EYE linear inserted between l1 and l2 should not change outputs.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    x = torch.randn(2, 4)
    y0 = gm(x)
    layer = LinearFactory.create_linear(4, 4, Layer_Type.EYE)

    # Act
    ModelStructureEditor.add_new_seq_layer(gm, "l1", "l2", layer, name="seq1")

    # Assert
    assert torch.allclose(gm(x), y0)


def test_add_new_seq_layer_eye_on_residual_skip_branch():
    """
    Sequential insert on l1→l4 should keep the residual graph valid and unchanged in output.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())
    x = torch.randn(2, 4)
    y0 = gm(x)
    layer = LinearFactory.create_linear(4, 4, Layer_Type.EYE)

    # Act
    ModelStructureEditor.add_new_seq_layer(gm, "l1", "l4", layer, name="seq_l1_l4")

    # Assert
    assert torch.allclose(gm(x), y0)


def test_delete_layer_removes_middle_layer_from_linear_chain():
    """
    delete_layer should bypass a middle linear and drop its submodule from gm.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    ModelStructureEditor.delete_layer(gm, "l2")
    module_names = [str(n.target) for n in gm.graph.nodes if n.op == "call_module"]

    # Assert
    assert module_names == ["l1", "l3"]
    assert not hasattr(gm, "l2")


def test_delete_layer_removes_branch_layer_from_residual_graph():
    """
    delete_layer on a branch module should leave other call_module nodes intact.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())

    # Act
    ModelStructureEditor.delete_layer(gm, "l2")
    module_names = [str(n.target) for n in gm.graph.nodes if n.op == "call_module"]

    # Assert
    assert module_names == ["l1", "l3", "l4"]
    assert not hasattr(gm, "l2")


def test_delete_layer_after_nary_add_residual_passes_lint():
    """
    delete_layer should rewire safely when the removed layer feeds a nary_add user.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", nn.Linear(4, 4), name="res1")

    # Act
    ModelStructureEditor.delete_layer(gm, "res1")

    # Assert
    gm.graph.lint()
    assert not hasattr(gm, "res1")


def test_add_new_seq_layer_raises_when_src_equals_dst():
    """
    add_new_seq_layer should reject identical src and dst endpoints.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act / Assert
    with pytest.raises(ValueError, match="src and dst must differ"):
        ModelStructureEditor.add_new_seq_layer(
            gm, "l1", "l1", nn.Linear(4, 4), name="seq_bad",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
