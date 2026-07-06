"""Unit tests for delete-layer shape helpers."""

import torch
import torch.fx as fx
import torch.nn as nn

from growingnn.actions.delete_layer import (
    DelLayer,
    can_bypass_delete_layer,
    get_common_input_shape,
    get_common_output_shape,
    has_same_input_shape,
    has_same_output_shape,
)
from growingnn.utils.fx import LayerBridgeFinder, LayerShapeAnalyser, ModelStructureEditor
from growingnn.utils.fx.graph_editor import bypass_shapes_compatible, compute_bypass_matching
from tests.model_factory import ModelFactory


def test_uniform_activation_shape_rejects_mismatched_rank2_shapes():
    """
    uniform_activation_shape should return None when layer shapes differ.
    """

    # Arrange / Act
    shape = LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 16)])

    # Assert
    assert shape is None


def test_has_same_output_shape_true_for_matching_probed_shapes():
    """
    has_same_output_shape should use LayerShapeAnalyser output shapes, not nn.Linear fields.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    output_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)

    # Act
    result = has_same_output_shape(gm, ["l1"], output_shapes=output_shapes)

    # Assert
    assert result is True
    assert get_common_output_shape(gm, ["l1"], output_shapes=output_shapes) == output_shapes["l1"]


def test_has_same_input_shape_false_when_successors_expect_different_shapes():
    """
    has_same_input_shape should be false when probed input shapes for successors differ.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    result = has_same_input_shape(
        gm,
        ["l2", "l3"],
        input_shapes={"l2": (1, 8), "l3": (1, 16)},
    )

    # Assert
    assert result is False


def test_del_layer_generate_finds_removable_middle_layer():
    """
    DelLayer.generate_all_actions should propose deleting a middle layer when bridge shapes match.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    actions = DelLayer.generate_all_actions(gm)

    # Assert
    assert any(action.params == ["l2"] for action in actions)


def test_has_same_input_shape_true_when_successors_share_input_shape():
    """
    has_same_input_shape should be true when all successor inputs share one probed shape.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    input_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm)

    # Act
    result = has_same_input_shape(gm, ["l3"], input_shapes)

    # Assert
    assert result is True


def test_get_common_output_shape_returns_shape_for_matching_predecessors():
    """
    get_common_output_shape should return the shared tuple when all inputs agree.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    output_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)

    # Act
    shape = get_common_output_shape(gm, ["l1"], output_shapes)

    # Assert
    assert shape == output_shapes["l1"]


def test_get_common_input_shape_returns_none_for_empty_layers():
    """
    get_common_input_shape should return None when no successor layers are given.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    shape = get_common_input_shape(gm, [])

    # Assert
    assert shape is None


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
    result = can_bypass_delete_layer(gm, "mid_a")

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
    result = can_bypass_delete_layer(gm, "res1")

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
    ModelStructureEditor.delete_layer(gm, "mid_a")
    y = gm(x)

    # Assert
    assert y.shape == (2, 4)
    assert not hasattr(gm, "mid_a")
