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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
