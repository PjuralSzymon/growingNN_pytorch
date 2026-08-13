"""Unit tests for ``growingnn.utils.fx.graph_analysis``."""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.fx import LayerBridgeFinder, LayerShapeAnalyser, ModuleClassifier, GraphStructureQuery, GraphConnectivity
from tests.model_factory import ModelFactory


def _linear_params(in_features: int, out_features: int, *, bias: bool = True) -> int:
    total = in_features * out_features
    if bias:
        total += out_features
    return total


def _conv2d_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    bias: bool = True,
) -> int:
    total = kernel_size * kernel_size * in_channels * out_channels
    if bias:
        total += out_channels
    return total


def _bn1d_params(num_features: int) -> int:
    return 2 * num_features


def test_uniform_activation_shape_returns_shared_shape_when_all_match():
    """
    uniform_activation_shape should return the common tuple when every shape is equal.
    """

    # Arrange / Act
    shape = LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 8)])

    # Assert
    assert shape == (1, 8)


def test_find_bridge_linear_sizes_maps_rank2_output_to_rank2_input():
    """
    find_bridge_linear_sizes should use the feature dim (last axis) for (batch, features).
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_bridge_linear_sizes((1, 8), (1, 16))

    # Assert
    assert sizes == (8, 16)


def test_find_bridge_res_linear_sizes_uses_rank2_outputs():
    """
    find_bridge_res_linear_sizes should read feature dims from both output tensors.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_bridge_res_linear_sizes((2, 4), (2, 8))

    # Assert
    assert sizes == (4, 8)


def test_find_conv_before_linear_sizes_sequential_keeps_channel_width():
    """
    Sequential conv-before-linear should use equal in/out channels on the conv.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_seq_conv_before_linear_sizes((1, 8, 7, 7), (1, 64))

    # Assert
    assert sizes == (8, 8)


def test_find_conv_before_linear_sizes_residual_uses_linear_output_width():
    """
    Residual conv-before-linear should map conv channels to linear output features.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_conv_before_linear_sizes(
        (1, 8, 7, 7), (1, 64), (1, 32), for_residual=True
    )

    # Assert
    assert sizes == (8, 32)


def test_find_equal_conv_output_shapes_requires_matching_rank4():
    """
    find_equal_conv_output_shapes should be true only for equal 4D shapes.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8, 7, 7), (1, 8, 7, 7)) is True
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8, 7, 7), (1, 16, 7, 7)) is False
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8), (1, 8)) is False


def test_find_bridge_linear_sizes_returns_none_for_conv_shapes():
    """
    find_bridge_linear_sizes should reject 4D conv activations, not flatten them.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_bridge_linear_sizes((1, 512, 7, 7), (1, 512, 7, 7)) is None


def test_find_bridge_linear_sizes_returns_none_when_shape_missing():
    """
    find_bridge_linear_sizes should return None if either shape is None.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_bridge_linear_sizes(None, (1, 4)) is None
    assert LayerBridgeFinder.find_bridge_linear_sizes((1, 4), None) is None


def test_find_bridge_res_linear_sizes_returns_none_when_shape_missing():
    """
    find_bridge_res_linear_sizes should return None if either output shape is None.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_bridge_res_linear_sizes(None, (1, 8)) is None
    assert LayerBridgeFinder.find_bridge_res_linear_sizes((1, 4), None) is None


def test_find_equal_conv_output_shapes_returns_false_when_shape_is_none():
    """
    find_equal_conv_output_shapes should be false when either argument is None.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_equal_conv_output_shapes(None, (1, 8, 7, 7)) is False
    assert LayerBridgeFinder.find_equal_conv_output_shapes((1, 8, 7, 7), None) is False


def test_find_conv_before_linear_sizes_returns_none_when_linear_features_not_multiple_of_channels():
    """
    find_conv_before_linear_sizes should reject linear inputs that are not divisible by conv channels.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_conv_before_linear_sizes(
        (1, 8, 7, 7), (1, 10), for_residual=False
    )

    # Assert
    assert sizes is None


def test_find_res_conv_before_linear_sizes_maps_channels_to_linear_output():
    """
    find_res_conv_before_linear_sizes should delegate to the residual conv-before-linear path.
    """

    # Arrange / Act
    sizes = LayerBridgeFinder.find_res_conv_before_linear_sizes(
        (1, 8, 7, 7), (1, 64), (1, 32)
    )

    # Assert
    assert sizes == (8, 32)


def test_find_seq_linear_after_conv_sizes_returns_none_for_invalid_shapes():
    """
    find_seq_linear_after_conv_sizes should return None when conv or linear shape is not usable.
    """

    # Arrange / Act / Assert
    assert LayerBridgeFinder.find_seq_linear_after_conv_sizes((1, 8), (1, 64)) is None


def test_find_seq_conv_bridge_channels_returns_channel_count_when_shapes_match():
    """
    find_seq_conv_bridge_channels should return channel width when from/to 4D shapes are equal.
    """

    # Arrange / Act
    channels = LayerBridgeFinder.find_seq_conv_bridge_channels((1, 16, 14, 14), (1, 16, 14, 14))

    # Assert
    assert channels == 16


def test_find_seq_conv_bridge_channels_returns_none_when_shapes_differ():
    """
    find_seq_conv_bridge_channels should return None when activation shapes do not match.
    """

    # Arrange / Act
    channels = LayerBridgeFinder.find_seq_conv_bridge_channels((1, 16, 14, 14), (1, 8, 14, 14))

    # Assert
    assert channels is None


def test_get_layer_output_shapes_records_conv_outputs():
    """
    After ShapeProp, get_layer_output_shapes should list a shape tuple for each
    traced Conv2d submodule (e.g. c1, c2).
    """

    # Arrange
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)
    x = torch.randn(1, 4, 16, 16)

    # Act
    shapes = LayerShapeAnalyser.get_layer_output_shapes(gm, x)

    # Assert
    assert "c1" in shapes and "c2" in shapes
    assert shapes["c1"] == (1, 4, 16, 16)
    assert shapes["c2"] == (1, 4, 16, 16)


def test_get_layer_input_shapes_matches_predecessor_output():
    """
    get_layer_input_shapes for c2 should match the output shape recorded for c1.
    """

    # Arrange
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)
    x = torch.randn(1, 4, 16, 16)

    # Act
    out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm, x)
    in_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm, x)

    # Assert
    assert in_shapes["c2"] == out_shapes["c1"]
    assert in_shapes["c1"] == (1, 4, 16, 16)


def test_module_dependency_pairs_linear_chain():
    """
    module_dependency_pairs on a 3-layer chain should list dependency into the first hidden pair.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    # Assert
    assert pairs == {("l1", "l2")}


def test_module_dependency_pairs_linear_chain_with_activation():
    """
    Activations between linears should not create spurious dependency pairs.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3_with_activation())

    # Act
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    # Assert
    assert pairs == {("l1", "l2")}


def test_module_dependency_pairs_deeply_nested_submodules():
    """
    Deeply nested submodule graphs should still yield dependency pairs.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.deeply_nested_submodules())

    # Act
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    # Assert
    assert len(pairs) == 10


def test_avoid_dependency_pairs_with_activation():
    """
    Adding activations should not change the count of dependency pairs on the residual model.
    """
    # Arrange
    gm_normal = fx.symbolic_trace(ModelFactory.complex_residual_many_widths())
    gm_activations = fx.symbolic_trace(ModelFactory.complex_residual_many_widths_with_activation())

    # Act
    pairs_normal = set(GraphStructureQuery.module_dependency_pairs(gm_normal))
    pairs_activations = set(GraphStructureQuery.module_dependency_pairs(gm_activations))

    # Assert
    assert len(pairs_normal) == len(pairs_activations)


def test_module_dependency_pairs_with_residual_skip():
    """
    Residual topology should expose transitive pairs including skip (l1, l4).
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())

    # Act
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    # Assert
    assert pairs == {
        ("l1", "l2"),
        ("l1", "l3"),
        ("l1", "l4"),
        ("l2", "l3"),
    }


def test_module_sequential_pairs_linear_chain():
    """
    module_sequential_pairs should list only immediate editable successors.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act / Assert
    assert set(GraphStructureQuery.module_sequential_pairs(gm)) == {("l1", "l2"), ("l2", "l3")}


def test_module_sequential_pairs_include_boundary_conv_to_linear():
    """
    module_sequential_pairs should connect boundary conv and linear when only they are editable.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_mnist_cnn())

    # Act
    pairs = set(GraphStructureQuery.module_sequential_pairs(gm))

    # Assert
    assert pairs == {("conv1", "linear")}


def test_module_sequential_pairs_with_residual_skip():
    """
    Sequential pairs on a residual graph follow the main path and branch merge.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.residual_skip())

    # Act / Assert
    assert set(GraphStructureQuery.module_sequential_pairs(gm)) == {
        ("l1", "l2"),
        ("l1", "l4"),
        ("l2", "l3"),
    }


def test_is_hidden_module_true_for_middle_module():
    """
    is_hidden_module should be true for a middle call_module in a chain.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    l2_node = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l2")

    # Act
    result = ModuleClassifier.is_hidden_module(l2_node)

    # Assert
    assert result is True


def test_is_edge_into_hidden_module_accepts_visible_or_hidden_to_hidden():
    """
    is_edge_into_hidden_module is true for visible→hidden and hidden→hidden only.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    l1 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l1")
    l2 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l2")
    l3 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l3")

    # Act / Assert
    assert ModuleClassifier.is_edge_into_hidden_module(l1, l2) is True
    assert ModuleClassifier.is_edge_into_hidden_module(l2, l3) is False
    assert ModuleClassifier.is_edge_into_hidden_module(l2, l1) is False
    assert ModuleClassifier.is_edge_into_hidden_module(l1, l3) is False


def test_get_all_hidden_modules_returns_only_hidden_linear_chain_modules():
    """
    get_all_hidden_modules on a 3-layer chain should list only the middle layer.
    """
    # Arrange
    model = ModelFactory.simple_chain_3()

    # Act
    result = GraphStructureQuery.get_all_hidden_modules(model)

    # Assert
    assert result == ["l2"]


def test_get_all_hidden_modules_returns_only_hidden_conv_chain_modules():
    """
    get_all_hidden_modules on a conv pipeline should list middle conv/pool/linear ids.
    """
    # Arrange
    model = ModelFactory.simple_conv_chain_2()

    # Act
    result = GraphStructureQuery.get_all_hidden_modules(model)

    # Assert
    assert result == ["c2", "pool", "l1"]


def test_is_editable_module_true_for_linear_call_module():
    """
    is_editable_module should accept nn.Linear call_module nodes on a traced graph.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    l1 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l1")

    # Act / Assert
    assert ModuleClassifier.is_editable_module(l1, gm) is True


def test_is_at_least_one_hidden_module_when_one_endpoint_is_hidden():
    """
    is_at_least_one_hidden_module should be true when either node is hidden.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    l1 = next(n for n in gm.graph.nodes if n.target == "l1")
    l2 = next(n for n in gm.graph.nodes if n.target == "l2")

    # Act / Assert
    assert ModuleClassifier.is_at_least_one_hidden_module(l1, l2) is True


def test_get_amount_of_parameters_single_linear_layer():
    """
    A traced Linear(4, 3) should contribute 4*3 weights plus 3 bias terms (15 total).
    """

    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x)

    gm = fx.symbolic_trace(Model())
    expected = _linear_params(4, 3)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_linear_chain_sums_each_layer():
    """
    Two linear layers (4->3 and 3->2) should sum to 15 + 8 = 23 parameters.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2_diffrent_input_output_features())
    expected = _linear_params(4, 3) + _linear_params(3, 2)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_ignores_relu_and_dropout():
    """
    ReLU, Dropout, and AdaptiveAvgPool1d have no weights; three linears and BatchNorm1d still count (68 total).
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3_with_activation())
    expected = (
        _linear_params(4, 4)
        + _linear_params(4, 4)
        + _linear_params(4, 4)
        + _bn1d_params(4)
    )

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_single_conv2d_layer():
    """
    Conv2d(2, 4, kernel_size=3) should contribute 3*3*2*4 weights plus 4 bias terms (76 total).
    """

    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    gm = fx.symbolic_trace(Model())
    expected = _conv2d_params(2, 4, 3)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_conv2d_without_bias():
    """
    Conv2d with bias=False should count only the kernel weights (3*3*1*2 = 18).
    """

    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, bias=False)

        def forward(self, x):
            return self.conv(x)

    gm = fx.symbolic_trace(Model())
    expected = _conv2d_params(1, 2, 3, bias=False)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_conv_plus_linear_head():
    """
    Conv2d(1, 2, 3) plus Linear(2, 3) should sum to 20 + 9 = 29 parameters.
    """

    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(2, 3)

        def forward(self, x):
            x = self.pool(self.conv(x))
            return self.fc(x.flatten(1))

    gm = fx.symbolic_trace(Model())
    expected = _conv2d_params(1, 2, 3) + _linear_params(2, 3)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_includes_batch_norm():
    """
    BatchNorm1d(4) adds 8 parameters (scale and bias) on top of two linear layers.
    """

    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4)
            self.l2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.l2(self.bn(self.l1(x)))

    gm = fx.symbolic_trace(Model())
    expected = _linear_params(4, 4) + _bn1d_params(4) + _linear_params(4, 2)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_counts_shared_module_once():
    """
    The same Linear submodule used twice in forward should be counted once (9 params, not 18).
    """

    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 3)

        def forward(self, x):
            return self.l(self.l(x))

    gm = fx.symbolic_trace(Model())
    expected = _linear_params(2, 3)

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_amount_of_parameters_full_conv_chain_matches_manual_sum():
    """
    simple_conv_chain_2 should match hand-counted conv and linear layers (pool has zero params).
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())
    expected = (
        _conv2d_params(4, 4, 3)
        + _conv2d_params(4, 4, 3)
        + _linear_params(4, 4)
        + _linear_params(4, 4)
    )

    # Act
    count = GraphStructureQuery.get_amount_of_parameters(gm)

    # Assert
    assert count == expected


def test_get_input_layers_and_output_layers_for_middle_layer():
    """
    get_input_layers / get_output_layers should return sequential neighbours of a hidden layer.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    inputs = GraphStructureQuery.get_input_layers("l2", gm)
    outputs = GraphStructureQuery.get_output_layers("l2", gm)

    # Assert
    assert inputs == ["l1"]
    assert outputs == ["l3"]


def test_path_has_dropout_detects_dropout_between_editable_pair():
    """
    path_has_dropout should be True after a Dropout sits between two Linear modules.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))
    assert GraphStructureQuery.path_has_dropout(gm, "l1", "l2") is False
    from growingnn.utils.fx import ModelStructureEditor

    ModelStructureEditor.add_new_seq_layer(gm, "l1", "l2", nn.Dropout(0.2), "seq_dropout_0")

    # Act
    result = GraphStructureQuery.path_has_dropout(gm, "l1", "l2")

    # Assert
    assert result is True


def test_node_shape_reads_shape_from_shapeprop_metadata():
    """
    node_shape should return a tuple after ShapeProp has populated node.meta.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    x = torch.randn(1, 4)
    LayerShapeAnalyser.collect_layer_shapes(gm, x)
    l1 = next(n for n in gm.graph.nodes if n.target == "l1")

    # Act
    shape = LayerShapeAnalyser.node_shape(l1)

    # Assert
    assert shape == (1, 4)


def test_make_probe_builds_tensor_from_input_shape():
    """
    make_probe should build a tensor with the given input shape on the graph device.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act
    probe = LayerShapeAnalyser.make_probe(gm, (1, 4))

    # Assert
    assert tuple(probe.shape) == (1, 4)


def test_input_shape_for_layer_reads_first_arg_node_shape():
    """
    input_shape_for_layer should mirror the output shape of the layer's first fx input.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    x = torch.randn(1, 4)
    LayerShapeAnalyser.collect_layer_shapes(gm, x)
    l2 = next(n for n in gm.graph.nodes if n.target == "l2")

    # Act
    in_shape = LayerShapeAnalyser.input_shape_for_layer(l2)

    # Assert
    assert in_shape == (1, 4)


def test_collect_layer_shapes_returns_output_and_input_maps():
    """
    collect_layer_shapes should return two dicts keyed by call_module target.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    x = torch.randn(1, 4)

    # Act
    outputs, inputs = LayerShapeAnalyser.collect_layer_shapes(gm, x)

    # Assert
    assert "l1" in outputs and "l2" in outputs
    assert "l2" in inputs


def test_linear_feature_dim_and_conv_channels():
    """
    linear_feature_dim and conv_channels should read the feature/channel axis only.
    """
    # Arrange / Act / Assert
    assert LayerBridgeFinder.linear_feature_dim((1, 8)) == 8
    assert LayerBridgeFinder.linear_feature_dim((1, 8, 7, 7)) is None
    assert LayerBridgeFinder.conv_channels((1, 8, 7, 7)) == 8
    assert LayerBridgeFinder.conv_channels((1, 8)) is None


def test_find_seq_conv_before_linear_sizes_matches_sequential_path():
    """
    find_seq_conv_before_linear_sizes should return equal channel width for valid 4D→2D pair.
    """
    # Arrange / Act
    sizes = LayerBridgeFinder.find_seq_conv_before_linear_sizes((1, 8, 7, 7), (1, 64))

    # Assert
    assert sizes == (8, 8)


def test_find_seq_linear_after_conv_sizes_returns_feature_dims():
    """
    find_seq_linear_after_conv_sizes should return (F, F) when conv 4D and linear 2D are valid.
    """
    # Arrange / Act
    sizes = LayerBridgeFinder.find_seq_linear_after_conv_sizes((1, 8, 4, 4), (1, 64))

    # Assert
    assert sizes == (64, 64)


def test_uniform_activation_shape_returns_none_when_shapes_differ():
    """
    uniform_activation_shape should return None when any shape in the list differs.
    """
    # Arrange / Act
    shape = LayerBridgeFinder.uniform_activation_shape([(1, 8), (1, 16)])

    # Assert
    assert shape is None


def test_graph_connectivity_detects_unreachable_modules_after_orphan_branch():
    """
    explain_connectivity should report dangling and unreachable modules on dead side branches.
    """

    # Arrange
    class DeadBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.orphan = nn.Linear(4, 4)
            self.head = nn.Linear(4, 4)

        def forward(self, x):
            h = self.stem(x)
            self.orphan(h)
            return self.head(h)

    gm = fx.symbolic_trace(DeadBranch())

    # Act
    issues = GraphConnectivity.explain_connectivity(gm)

    # Assert
    assert GraphConnectivity.is_connected_to_output(gm) is False
    assert GraphConnectivity.unreachable_module_ids(gm) == ["orphan"]
    assert any("dangling leaves" in issue for issue in issues)
    assert any("unreachable modules" in issue for issue in issues)


def test_prune_unreachable_nodes_removes_dangling_branch():
    """
    prune_unreachable_nodes should erase side branches that no longer reach the graph output.
    """

    # Arrange
    class DeadBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(4, 4)
            self.orphan = nn.Linear(4, 4)
            self.head = nn.Linear(4, 4)

        def forward(self, x):
            h = self.stem(x)
            self.orphan(h)
            return self.head(h)

    gm = fx.symbolic_trace(DeadBranch())

    # Act
    from growingnn.utils.fx.graph_editor import prune_unreachable_nodes
    removed = prune_unreachable_nodes(gm)

    # Assert
    assert "orphan" in removed
    assert not hasattr(gm, "orphan")
    assert GraphConnectivity.is_connected_to_output(gm)
    assert GraphConnectivity.live_module_ids(gm) == ["stem", "head"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
