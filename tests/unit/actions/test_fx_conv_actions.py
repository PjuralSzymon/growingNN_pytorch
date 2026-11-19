import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import pair_traversal_decision


class _LargeSpatialGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1)
        self.ops = nn.ModuleList(nn.ReLU() for _ in range(300))
        self.left = nn.Conv2d(4, 4, kernel_size=1)
        self.right = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, value):
        value = self.stem(value)
        for op in self.ops:
            value = op(value)
        return self.left(value) + self.right(value)


def test_large_spatial_graph_uses_fx_conv_edge_and_merge_actions():
    torch.manual_seed(31)
    model = _LargeSpatialGraph().eval()
    value = torch.randn(1, 3, 8, 8)
    traced = TracedModel.create(model, (1, 3, 8, 8))
    original = traced.gm(value)

    decision = pair_traversal_decision(traced.gm)
    assert not decision.use_legacy_pairs
    assert AddSeqConvLayer.generate_all_actions(traced) == []
    assert AddResConvLayer.generate_all_actions(traced) == []

    sequential = AddSeqConvLayer.generate_all_actions(traced, feature_axis=1)
    residual = AddResConvLayer.generate_all_actions(traced, feature_axis=1)
    assert sequential and all(action.params[4] == "fx_edge" for action in sequential)
    assert residual and all(action.params[4] == "fx_residual" for action in residual)

    sequential_graph = copy.deepcopy(traced.gm)
    sequential[0].execute(TracedModel.create(sequential_graph, (1, 3, 8, 8)))
    torch.testing.assert_close(sequential_graph(value), original)

    residual_graph = copy.deepcopy(traced.gm)
    residual[0].execute(TracedModel.create(residual_graph, (1, 3, 8, 8)))
    torch.testing.assert_close(residual_graph(value), original)


def test_feature_axis_conv_adapter_preserves_nhwc_shape_and_identity():
    from growingnn.utils.fx import FeatureAxisConv2dAdapter

    value = torch.randn(2, 7, 9, 4)
    adapter = FeatureAxisConv2dAdapter(4, feature_axis=-1, residual=True)

    actual = adapter(value)

    assert actual.shape == value.shape
    torch.testing.assert_close(actual, value)


def test_large_spatial_graph_rejects_batch_axis_as_conv_feature_axis():
    traced = TracedModel.create(_LargeSpatialGraph().eval(), (1, 3, 8, 8))

    assert AddSeqConvLayer.generate_all_actions(traced, feature_axis=0) == []
    assert AddResConvLayer.generate_all_actions(traced, feature_axis=0) == []


def test_functional_convolution_nodes_form_action_boundaries_without_modules():
    class FunctionalConvGraph(nn.Module):
        def __init__(self):
            super().__init__()
            self.first_weight = nn.Parameter(torch.randn(4, 4, 1, 1))
            self.left_weight = nn.Parameter(torch.randn(4, 4, 1, 1))
            self.right_weight = nn.Parameter(torch.randn(4, 4, 1, 1))

        def forward(self, value):
            value = F.conv2d(value, self.first_weight)
            for _ in range(40):
                value = F.relu(value)
            left = F.conv2d(value, self.left_weight)
            right = F.conv2d(value, self.right_weight)
            return left + right

    torch.manual_seed(37)
    model = FunctionalConvGraph().eval()
    value = torch.randn(1, 4, 8, 8)
    traced = TracedModel.create(model, tuple(value.shape), example_args=(value,))
    original = traced.gm(value)

    assert not any(node.op == "call_module" for node in traced.gm.graph.nodes)
    sequential = AddSeqConvLayer.generate_all_actions(traced, feature_axis=1)
    residual = AddResConvLayer.generate_all_actions(traced, feature_axis=1)
    assert sequential and all(action.params[4] == "fx_edge" for action in sequential)
    assert residual and all(action.params[4] == "fx_residual" for action in residual)

    sequential_graph = copy.deepcopy(traced.gm)
    sequential[0].execute(TracedModel.create(sequential_graph, tuple(value.shape)))
    torch.testing.assert_close(sequential_graph(value), original)

    residual_graph = copy.deepcopy(traced.gm)
    residual[0].execute(TracedModel.create(residual_graph, tuple(value.shape)))
    torch.testing.assert_close(residual_graph(value), original)
