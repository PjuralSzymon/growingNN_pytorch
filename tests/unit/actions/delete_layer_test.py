"""Unit tests for delete-layer shape helpers."""

import torch
import torch.fx as fx
import torch.nn as nn
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from growingnn.actions.delete_layer import DelLayer, can_bypass_delete_layer
from growingnn.utils.fx import LayerBridgeFinder, LayerShapeAnalyser, ModelStructureEditor
from growingnn.utils.fx.graph_editor import compute_bypass_matching
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


def _uniform_layer_shape(
    shape_map: dict[str, tuple[int, ...]],
    layer_ids: list[str],
) -> tuple[int, ...] | None:
    shapes = [shape_map.get(layer_id) for layer_id in layer_ids]
    return LayerBridgeFinder.uniform_activation_shape(shapes)


def test_uniform_activation_shape_rejects_mismatched_rank2_shapes():
    """
    uniform_activation_shape should return None when layer shapes differ.
    """

    # Arrange / Act
    shape = LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 16)])

    # Assert
    assert shape is None


def test_uniform_layer_shape_true_for_matching_probed_outputs():
    """
    Probed output shapes for predecessors should share one activation tuple when widths match.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    output_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm, input_shape=(1, 4))

    # Act
    shape = _uniform_layer_shape(output_shapes, ["l1"])

    # Assert
    assert shape == output_shapes["l1"]


def test_uniform_layer_shape_false_when_successor_inputs_differ():
    """
    uniform_layer_shape should be None when probed successor input shapes differ.
    """

    # Arrange / Act
    shape = _uniform_layer_shape({"l2": (1, 8), "l3": (1, 16)}, ["l2", "l3"])

    # Assert
    assert shape is None


def test_uniform_layer_shape_true_when_successor_inputs_agree():
    """
    uniform_layer_shape should return the shared tuple when successor inputs agree.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    input_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm, input_shape=(1, 4))

    # Act
    shape = _uniform_layer_shape(input_shapes, ["l3"])

    # Assert
    assert shape == input_shapes["l3"]


def test_uniform_layer_shape_returns_none_for_empty_layers():
    """
    uniform_layer_shape should return None when no layer ids are given.
    """

    # Arrange / Act
    shape = _uniform_layer_shape({}, [])

    # Assert
    assert shape is None


def test_del_layer_generate_finds_removable_middle_layer():
    """
    DelLayer.generate_all_actions should propose deleting a middle layer when bridge shapes match.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    actions = DelLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))

    # Assert
    assert any(action.params == ["l2"] for action in actions)


def test_compute_bypass_matching_pairs_different_predecessor_shapes():
    """
    compute_bypass_matching should pair predecessors to successors individually instead of requiring one uniform width.
    """

    # Arrange
    output_shapes = {"a": (1, 8), "b": (1, 16)}
    input_shapes = {"ca": (1, 8), "cb": (1, 16)}

    # Act
    matching = compute_bypass_matching(["a", "b"], ["ca", "cb"], output_shapes, input_shapes)

    # Assert
    assert matching == {"ca": "a", "cb": "b"}
    assert LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 16)]) is None


def test_can_bypass_delete_layer_true_for_pairwise_branch_mids():
    """
    can_bypass_delete_layer should allow deleting one branch middle layer without requiring all predecessors to share one width.
    """

    # Arrange
    class PairwiseBranches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 8)
            self.b = nn.Linear(4, 16)
            self.mid_a = nn.Linear(8, 8)
            self.mid_b = nn.Linear(16, 16)
            self.ca = nn.Linear(8, 4)
            self.cb = nn.Linear(16, 4)

        def forward(self, x):
            return self.ca(self.mid_a(self.a(x))) + self.cb(self.mid_b(self.b(x)))

    gm = fx.symbolic_trace(PairwiseBranches())

    # Act
    result = can_bypass_delete_layer(gm, "mid_a", input_shape=(1, 4))

    # Assert
    assert result is True


def test_can_bypass_delete_layer_true_for_merge_branch_residual():
    """
    Residual branches that only feed nary_add should be deletable without uniform bypass shapes.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    ModelStructureEditor.add_new_residual_layer(gm, "l1", "l2", nn.Linear(4, 4), name="res1")

    # Act
    result = can_bypass_delete_layer(gm, "res1", input_shape=(1, 4))

    # Assert
    assert result is True


def test_delete_pairwise_branch_mid_preserves_forward_shape():
    """
    delete_layer should bypass only the compatible predecessor branch instead of summing every input.
    """

    # Arrange
    class PairwiseBranches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 8)
            self.b = nn.Linear(4, 16)
            self.mid_a = nn.Linear(8, 8)
            self.mid_b = nn.Linear(16, 16)
            self.ca = nn.Linear(8, 4)
            self.cb = nn.Linear(16, 4)

        def forward(self, x):
            return self.ca(self.mid_a(self.a(x))) + self.cb(self.mid_b(self.b(x)))

    gm = fx.symbolic_trace(PairwiseBranches())
    x = torch.randn(2, 4)

    # Act
    ModelStructureEditor.delete_layer(gm, "mid_a", input_shape=(1, 4))
    y = gm(x)

    # Assert
    assert y.shape == (2, 4)
    assert not hasattr(gm, "mid_a")
