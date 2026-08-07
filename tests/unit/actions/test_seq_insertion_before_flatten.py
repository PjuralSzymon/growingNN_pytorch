"""Unit tests for before-flatten sequential-convolution spatial legality helpers."""

import sys
from pathlib import Path

import pytest
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_analysis import GraphStructureQuery, LayerBridgeFinder
from growingnn.utils.fx.graph_editor import _path_dst_to_src
from growingnn.utils.fx.node_analysis import ModuleResolver, NodeTypeChecker
from experiments.train_mnist_exp001_slope_model_depth import MediumMnistNet
from tests.model_factory import ModelFactory


def test_get_convolution_output_height_and_width_preserves_one_by_one_with_padding():
    """
    3x3 conv with pad=1 on 1x1 spatial should stay 1x1.
    """
    # Arrange / Act
    result = LayerBridgeFinder.get_convolution_output_height_and_width(
        1, 1, kernel_size=3, stride=1, padding=1,
    )

    # Assert
    assert result == (1, 1)


def test_get_flatten_feature_count_accepts_medium_insert_site():
    """
    Medium-like site (C=4,1,1) with eye 3x3 pad1 should yield flatten features 4.
    """
    # Arrange / Act
    feature_count = LayerBridgeFinder.get_flatten_feature_count_after_convolution_and_pools(
        insert_site_channels=4,
        insert_site_height=1,
        insert_site_width=1,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        pools_after_insert=[],
    )

    # Assert
    assert feature_count == 4


def test_get_flatten_feature_count_rejects_spatial_mismatch_without_later_pool():
    """
    Site (4,2,2) with no later pool should yield 16, not match linear_in=4.
    """
    # Arrange / Act
    feature_count = LayerBridgeFinder.get_flatten_feature_count_after_convolution_and_pools(
        insert_site_channels=4,
        insert_site_height=2,
        insert_site_width=2,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        pools_after_insert=[],
    )

    # Assert
    assert feature_count == 16
    assert feature_count != 4


def test_get_flatten_feature_count_with_adaptive_pool_after_insert_uses_out_channels():
    """
    Adaptive-to-1 after insert should make flatten feature count equal out_channels.
    """
    # Arrange / Act
    feature_count = LayerBridgeFinder.get_flatten_feature_count_after_convolution_and_pools(
        insert_site_channels=8,
        insert_site_height=7,
        insert_site_width=7,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        pools_after_insert=[("adaptive_avg", {"output_size": 1})],
    )

    # Assert
    assert feature_count == 8


def test_medium_path_has_pools_and_method_flatten():
    """
    MediumMnistNet path conv1→linear should expose pools before method flatten.
    """
    # Arrange
    traced = TracedModel.create(MediumMnistNet(), (1, 1, 28, 28))
    traced.shapes()
    gm = traced.gm
    src = ModuleResolver.find_call_module(gm.graph.nodes, "conv1")
    dst = ModuleResolver.find_call_module(gm.graph.nodes, "linear")
    path = _path_dst_to_src(dst, src)

    # Act
    flatten_node = GraphStructureQuery.find_flatten_node_on_path_toward_source(path, gm)
    pools = GraphStructureQuery.find_pool_nodes_between_flatten_and_source(path, flatten_node, gm)

    # Assert
    assert flatten_node is not None
    assert flatten_node.op == "call_method" and flatten_node.target == "flatten"
    assert len(pools) >= 1
    assert any(NodeTypeChecker.two_d_pool_kind(p, gm) == "adaptive_avg" for p in pools)


def test_is_before_flatten_insert_accepts_medium_conv_to_linear():
    """
    Insert check should be True for MediumMnistNet conv1→linear; then get eye-conv shape.
    """
    # Arrange
    traced = TracedModel.create(MediumMnistNet(), (1, 1, 28, 28))

    # Act
    can_insert = AddSeqConvLayer.is_before_flatten_insert(traced, "conv1", "linear")
    eye_shape = AddSeqConvLayer.get_eye_convolution_shape_for_before_flatten(
        traced, "conv1", "linear",
    )

    # Assert
    assert can_insert is True
    assert eye_shape is not None
    out_channels, kernel_size, padding = eye_shape
    layer = ConvFactory.create_eye_conv(
        out_channels, out_channels, kernel_size, stride=1, padding=padding,
    )
    assert layer.in_channels == 4
    assert layer.out_channels == 4


def test_get_eye_convolution_shape_uses_one_by_one_fallback_for_unpadded_source():
    """
    When source 3x3 has no padding on a 1x1 map, shape should use kernel 1 fallback.
    """
    # Arrange
    traced = TracedModel.create(ModelFactory.simple_mnist_cnn(), (1, 1, 28, 28))

    # Act
    assert AddSeqConvLayer.is_before_flatten_insert(traced, "conv1", "linear")
    eye_shape = AddSeqConvLayer.get_eye_convolution_shape_for_before_flatten(
        traced, "conv1", "linear",
    )

    # Assert
    assert eye_shape is not None
    _, kernel_size, padding = eye_shape
    assert kernel_size == 1
    assert padding == 0


def test_is_before_flatten_insert_rejects_linear_to_linear():
    """
    Insert check should be False for linear→linear pairs.
    """
    # Arrange
    traced = TracedModel.create(MediumMnistNet(), (1, 1, 28, 28))

    # Act
    can_insert = AddSeqConvLayer.is_before_flatten_insert(traced, "linear", "linear2")

    # Assert
    assert can_insert is False


def test_is_before_flatten_insert_rejects_classifier_past_hidden_linear():
    """
    Big conv2→linear2 should be rejected because hidden linear sits after flatten.
    """
    # Arrange
    from experiments.train_mnist_exp002_initial_architectures import BigAvgPoolMnistNet

    traced = TracedModel.create(BigAvgPoolMnistNet(), (1, 1, 28, 28))

    # Act
    can_insert = AddSeqConvLayer.is_before_flatten_insert(traced, "conv2", "linear2")

    # Assert
    assert can_insert is False


def test_generate_all_actions_on_big_does_not_raise():
    """
    Action generation on the big MNIST starter should stay stable during search.
    """
    # Arrange
    from experiments.train_mnist_exp002_initial_architectures import BigAvgPoolMnistNet

    traced = TracedModel.create(BigAvgPoolMnistNet(), (1, 1, 28, 28))

    # Act
    actions = AddSeqConvLayer.generate_all_actions(traced)

    # Assert
    assert len(actions) >= 1
    assert all(action.params[1] != "linear2" for action in actions)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
