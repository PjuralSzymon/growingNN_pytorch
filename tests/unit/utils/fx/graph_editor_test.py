"""Unit tests for ``growingnn.utils.fx.graph_editor``."""

import sys
from pathlib import Path

import pytest
import operator

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
    ModelStructureEditor.delete_layer(gm, "l2", input_shape=(1, 4))
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
    ModelStructureEditor.delete_layer(gm, "l2", input_shape=(1, 4))
    module_names = [str(n.target) for n in gm.graph.nodes if n.op == "call_module"]

    # Assert
    assert module_names == ["l1", "l3", "l4"]
    assert not hasattr(gm, "l2")


def test_delete_layer_after_nary_add_residual_passes_lint():
    """
    delete_layer should remove a residual branch from nary_add without rewiring its source into the sum.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", nn.Linear(4, 4), name="res1")

    # Act
    ModelStructureEditor.delete_layer(gm, "res1", input_shape=(1, 4))

    # Assert
    gm.graph.lint()
    assert not hasattr(gm, "res1")
    nary_nodes = [node for node in gm.graph.nodes if node.op == "call_function" and node.target is nary_add]
    assert len(nary_nodes) == 0


def test_delete_merge_branch_residual_eye_preserves_forward():
    """
    A residual EYE branch that only feeds nary_add should be removable without shape bypass.
    """

    # Arrange
    class ResEyeBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 4)
            self.l3 = nn.Linear(4, 4)

        def forward(self, x):
            return self.l3(self.l2(self.l1(x)) + self.l2(self.l1(x)))

    gm = fx.symbolic_trace(ResEyeBranch())
    eye = nn.Linear(4, 4)
    eye.weight.data.zero_()
    eye.bias.data.zero_()
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", eye, name="res1")
    x = torch.randn(2, 4)
    y_before = gm(x)

    # Act
    ModelStructureEditor.delete_layer(gm, "res1", input_shape=(1, 4))
    y_after = gm(x)

    # Assert
    assert not hasattr(gm, "res1")
    assert torch.allclose(y_before, y_after)


def test_delete_layer_prunes_unreachable_trunk_branches():
    """
    delete_layer should remove orphaned subgraphs that no longer reach the graph output.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", nn.Linear(4, 4), name="res1")
    x = torch.randn(2, 4)

    # Act
    ModelStructureEditor.delete_layer(gm, "res1", input_shape=(1, 4))
    y = gm(x)

    # Assert
    from growingnn.utils.fx import GraphConnectivity
    assert GraphConnectivity.is_connected_to_output(gm)
    assert y.shape == (2, 4)
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


def test_insert_module_on_edge_supports_function_and_method_nodes():
    class FunctionalModel(nn.Module):
        def forward(self, x):
            activated = torch.relu(x)
            return activated.reshape(activated.shape[0], -1)

    gm = fx.symbolic_trace(FunctionalModel())
    x = torch.randn(2, 3, 4)
    expected = gm(x)
    source = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is torch.relu)
    destination = next(node for node in gm.graph.nodes if node.op == "call_method" and node.target == "reshape")

    inserted = ModelStructureEditor.insert_module_on_edge(
        gm, source, destination, nn.Identity(), "edge_identity"
    )

    assert inserted.op == "call_module"
    assert inserted.args == (source,)
    torch.testing.assert_close(gm(x), expected)


def test_insert_module_on_edge_rejects_non_edge():
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    nodes = list(gm.graph.nodes)
    source = next(node for node in nodes if node.op == "call_module" and node.target == "l1")
    destination = next(node for node in nodes if node.op == "output")

    with pytest.raises(ValueError, match="non-edge"):
        ModelStructureEditor.insert_module_on_edge(gm, source, destination, nn.Identity(), "bad_edge")


def test_insert_module_on_repeated_edge_requires_and_honors_operand_path():
    class RepeatedUse(nn.Module):
        def forward(self, x):
            return x + x

    gm = fx.symbolic_trace(RepeatedUse())
    value = torch.randn(2, 3)
    expected = gm(value)
    x = next(node for node in gm.graph.nodes if node.op == "placeholder")
    add = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is operator.add)
    with pytest.raises(ValueError, match="2 operand uses"):
        ModelStructureEditor.insert_module_on_edge(gm, x, add, nn.Identity(), "ambiguous")

    inserted = ModelStructureEditor.insert_module_on_edge(
        gm, x, add, nn.Identity(), "one_use", use_path=("args", 0)
    )
    assert add.args == (inserted, x)
    torch.testing.assert_close(gm(value), expected)


def test_insert_module_on_edge_rejects_invalid_operand_path():
    class RepeatedUse(nn.Module):
        def forward(self, x):
            return x + x

    gm = fx.symbolic_trace(RepeatedUse())
    x = next(node for node in gm.graph.nodes if node.op == "placeholder")
    add = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is operator.add)
    with pytest.raises(ValueError, match="not an occurrence"):
        ModelStructureEditor.insert_module_on_edge(
            gm, x, add, nn.Identity(), "invalid_path", use_path=("args", 2)
        )


def test_insert_module_on_edge_rolls_back_if_operand_rewrite_fails(monkeypatch):
    class FunctionalModel(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    import growingnn.utils.fx.graph_editor as graph_editor

    gm = fx.symbolic_trace(FunctionalModel())
    value = torch.randn(2, 3)
    expected = gm(value)
    source = next(node for node in gm.graph.nodes if node.op == "placeholder")
    destination = next(node for node in gm.graph.nodes if node.op == "call_function")
    original_nodes = tuple(gm.graph.nodes)

    def fail_replacement(*args, **kwargs):
        raise ValueError("custom argument container cannot be rebuilt")

    monkeypatch.setattr(graph_editor, "_replace_at_path", fail_replacement)
    with pytest.raises(ValueError, match="cannot be rebuilt"):
        ModelStructureEditor.insert_module_on_edge(
            gm, source, destination, nn.Identity(), "rollback_layer"
        )

    assert tuple(gm.graph.nodes) == original_nodes
    assert not hasattr(gm, "rollback_layer")
    torch.testing.assert_close(gm(value), expected)


def test_fx_insert_uses_source_dtype_in_mixed_dtype_graph():
    class MixedDtype(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x).to(dtype=torch.bfloat16)

    gm = fx.symbolic_trace(MixedDtype()).eval()
    value = torch.randn(2, 4)
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(gm).propagate(value)
    source = next(node for node in gm.graph.nodes if node.op == "call_method" and node.target == "to")
    output = next(node for node in gm.graph.nodes if node.op == "output")
    layer = nn.Linear(4, 4)

    ModelStructureEditor.insert_module_on_edge(gm, source, output, layer, "bf16_edge")

    assert gm.bf16_edge.weight.dtype is torch.bfloat16
    assert gm.bf16_edge.weight.device == gm.proj.weight.device
    assert gm(value).dtype is torch.bfloat16


def test_insert_module_on_one_occurrence_inside_slice_argument():
    class SliceModel(nn.Module):
        def forward(self, x, index):
            return x[index:index]

    gm = fx.symbolic_trace(SliceModel())
    x_value = torch.arange(8)
    index_value = 2
    expected = gm(x_value, index_value)
    index = next(node for node in gm.graph.nodes if node.op == "placeholder" and node.name == "index")
    getitem = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is operator.getitem)

    inserted = ModelStructureEditor.insert_module_on_edge(
        gm, index, getitem, nn.Identity(), "slice_bound", use_path=("args", 1, "start")
    )

    assert getitem.args[1].start is inserted
    assert getitem.args[1].stop is index
    torch.testing.assert_close(gm(x_value, index_value), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
