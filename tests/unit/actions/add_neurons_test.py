"""Unit tests for ``growingnn.actions.add_neurons`` neuron-grow propagation."""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_neurons import AddNeurons, expand_layer_output
from growingnn.actions.utils.layer_resize import (
    can_resize_linear_output,
    linear_output_resize_rejection_reason,
)
from growingnn.core import config
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


def test_add_neurons_can_grow_linear_through_functional_pointwise_path():
    class PointwiseProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand = nn.Linear(4, 6)
            self.project = nn.Linear(6, 3)

        def forward(self, x):
            hidden = self.expand(x)
            hidden = (hidden * 0.5).pow(3) + torch.tanh(hidden)
            return self.project(hidden)

    gm = fx.symbolic_trace(PointwiseProjection())
    example = torch.randn(2, 4)
    original_shape = gm(example).shape

    assert can_resize_linear_output(gm, "expand", 9)
    AddNeurons(["expand", 1.5]).execute(TracedModel.create(gm, (2, 4)))

    assert gm.expand.out_features == 9
    assert gm.project.in_features == 9
    assert gm(example).shape == original_shape


def test_large_projection_growth_uses_model_relative_parameter_budget(monkeypatch):
    class LargeProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand = nn.Linear(768, 3072)

        def forward(self, x):
            return self.expand(x)

    gm = fx.symbolic_trace(LargeProjection())
    monkeypatch.setattr(config, "MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE", 100)
    monkeypatch.setattr(config, "MAX_MODEL_RELATIVE_NEURON_GROWTH_FRACTION", 0.2)
    monkeypatch.setattr(config, "MIN_MODEL_PARAMETER_COUNT_FOR_RELATIVE_NEURON_GROWTH", 1_000)

    assert can_resize_linear_output(gm, "expand", 3379)
    assert not can_resize_linear_output(gm, "expand", 4608)


def test_exported_functional_linear_chain_generates_and_executes_width_growth():
    class FunctionalMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 6)
            self.down = nn.Linear(6, 3)

        def forward(self, value):
            return self.down(torch.nn.functional.gelu(self.up(value)))

    example = torch.randn(2, 4)
    model = FunctionalMlp().eval()
    from growingnn.utils.fx import capture_graph, describe_linear_operations

    gm = capture_graph(model, route="export", example_args=(example,))
    traced = TracedModel.create(gm, (2, 4), example_args=(example,))
    original = gm(example)
    actions = AddNeurons.generate_all_actions(traced, ratio=1.5)

    functional_actions = [action for action in actions if len(action.params) > 2]
    assert len(functional_actions) == 1
    assert functional_actions[0].params[2] == "fx_functional"
    functional_actions[0].execute(traced)

    projections = [item for item in describe_linear_operations(gm) if item.form == "function"]
    assert len(projections) == 2
    assert [(item.in_features, item.out_features) for item in projections] == [(4, 9), (9, 3)]
    output = gm(example)
    assert output.shape == original.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in gm.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in gm.parameters())


def test_exported_gated_functional_mlp_generates_one_grouped_width_growth():
    class GatedMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 6)
            self.up = nn.Linear(4, 6)
            self.down = nn.Linear(6, 3)

        def forward(self, value):
            return self.down(torch.nn.functional.silu(self.gate(value)) * self.up(value))

    from growingnn.utils.fx import capture_graph, describe_linear_operations

    example = torch.randn(2, 4)
    gm = capture_graph(GatedMlp().eval(), route="export", example_args=(example,))
    traced = TracedModel.create(gm, (2, 4), example_args=(example,))
    actions = AddNeurons.generate_all_actions(traced, ratio=1.5)
    functional_actions = [action for action in actions if len(action.params) > 2]

    assert len(functional_actions) == 1
    functional_actions[0].execute(traced)

    projections = [item for item in describe_linear_operations(gm) if item.form == "function"]
    widths = {(item.in_features, item.out_features) for item in projections}
    assert widths == {(4, 9), (9, 3)}
    output = gm(example)
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in gm.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in gm.parameters())


def test_gated_functional_resize_rejects_fanout_on_other_projection_branch():
    class FanoutGatedMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 6)
            self.up = nn.Linear(4, 6)
            self.down = nn.Linear(6, 3)

        def forward(self, value):
            up = self.up(value)
            return self.down(torch.nn.functional.silu(self.gate(value)) * up), up

    from growingnn.utils.fx import (
        capture_graph,
        describe_linear_operations,
        functional_linear_resize_plan,
    )

    example = torch.randn(2, 4)
    gm = capture_graph(FanoutGatedMlp().eval(), route="export", example_args=(example,))
    projection = next(
        item for item in describe_linear_operations(gm)
        if item.form == "function" and item.out_features == 6
    )
    plan, reason = functional_linear_resize_plan(gm, projection.node, 9)

    assert plan is None
    assert reason is not None and "fanout" in reason.lower()


def test_functional_linear_width_growth_rejects_split_boundary_with_reason():
    class SplitFunctionalProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 12)

        def forward(self, value):
            return torch.split(self.projection(value), 4, dim=-1)

    from growingnn.utils.fx import (
        capture_graph,
        describe_linear_operations,
        functional_linear_resize_plan,
    )

    example = torch.randn(2, 4)
    gm = capture_graph(
        SplitFunctionalProjection().eval(), route="export", example_args=(example,)
    )
    traced = TracedModel.create(gm, (2, 4), example_args=(example,))
    traced.shapes()
    projection = next(
        item for item in describe_linear_operations(gm) if item.form == "function"
    )

    plan, reason = functional_linear_resize_plan(gm, projection.node, 18)

    assert plan is None
    assert reason is not None and "split" in reason and "layout changes" in reason


def test_functional_linear_width_growth_rejects_learned_prelu_path():
    class PreluMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 6)
            self.activation = nn.PReLU(6)
            self.down = nn.Linear(6, 3)

        def forward(self, value):
            return self.down(self.activation(self.up(value)))

    from growingnn.utils.fx import (
        capture_graph,
        describe_linear_operations,
        functional_linear_resize_plan,
    )

    example = torch.randn(2, 4)
    gm = capture_graph(PreluMlp().eval(), route="export", example_args=(example,))
    traced = TracedModel.create(gm, (2, 4), example_args=(example,))
    traced.shapes()
    projection = next(
        item for item in describe_linear_operations(gm) if item.form == "function"
    )

    plan, reason = functional_linear_resize_plan(gm, projection.node, 9)

    assert plan is None
    assert reason is not None and "prelu" in reason.lower()


def test_projection_resize_rejects_uncoordinated_split_boundary():
    class SplitProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 12)

        def forward(self, x):
            query, key, value = torch.split(self.qkv(x), 4, dim=-1)
            return query + key + value

    gm = fx.symbolic_trace(SplitProjection())

    assert not can_resize_linear_output(gm, "qkv", 18)
    reason = linear_output_resize_rejection_reason(gm, "qkv", 18)
    assert reason is not None and "feature layout" in reason and "split" in reason


def test_expand_linear_chain_updates_downstream_input():
    """
    Growing a hidden linear output should update the next linear in_features.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))
    x = torch.randn(2, 4)

    # Act
    expand_layer_output(gm, "l2", 2.0)
    y = gm(x)

    # Assert
    assert gm.l2.out_features == 20
    assert gm.l3.in_features == 20
    assert y.shape == (2, 10)


def test_expand_layer_output_noop_when_ratio_does_not_increase_width():
    """
    expand_layer_output should leave the graph unchanged when ratio does not increase width.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))
    before = gm.l2.out_features

    # Act
    expand_layer_output(gm, "l2", 1.0)

    # Assert
    assert gm.l2.out_features == before


def test_expand_layer_output_noop_when_matrix_limit_exceeded(monkeypatch):
    """
    expand_layer_output should noop when grow would exceed MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE.
    """
    # Arrange
    monkeypatch.setattr(config, "MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE", 1_000_000)
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=1000))
    before = gm.l2.out_features

    # Act
    expand_layer_output(gm, "l2", 1.5)

    # Assert
    assert gm.l2.out_features == before


def test_generate_actions_skips_layer_when_matrix_limit_exceeded(monkeypatch):
    """
    AddNeurons.generate_all_actions should omit layers whose grow exceeds the matrix size cap.
    """
    # Arrange
    monkeypatch.setattr(config, "MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE", 1_000_000)
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=1000))

    # Act
    actions = AddNeurons.generate_all_actions(TracedModel.create(gm, (1, 3, 32, 32)))

    # Assert
    assert all(action.params[0] != "l2" for action in actions)


def test_generate_actions_allows_linear_with_conv_sibling_at_add():
    """
    AddNeurons.generate_all_actions should include linears on conv/add residual paths.
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

    # Act
    actions = AddNeurons.generate_all_actions(TracedModel.create(gm, (1, 3, 32, 32)))

    # Assert
    layer_ids = [action.params[0] for action in actions]
    assert "linear_hidden" in layer_ids


def test_grow_linear_with_conv_sibling_resizes_conv_output():
    """
    Growing linear_hidden should resize the conv sibling output channels at the add.
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
    x = torch.randn(2, 3, 8, 8)

    # Act
    AddNeurons(["linear_hidden", 1.5]).execute(TracedModel.create(gm, (1, 3, 32, 32)))
    y = gm(x)

    # Assert
    assert gm.linear_hidden.out_features == 15
    assert gm.conv.out_channels == 15
    assert gm.fc.in_features == 15
    assert y.shape == (2, 5)


def test_add_neurons_execute_grows_hidden_layer():
    """
    AddNeurons.execute should widen a hidden linear and keep the model runnable.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))
    x = torch.randn(2, 4)

    # Act
    AddNeurons(["l2", 1.5]).execute(TracedModel.create(gm, (1, 3, 32, 32)))
    y = gm(x)

    # Assert
    assert gm.l2.out_features == 15
    assert gm.l3.in_features == 15
    assert y.shape == (2, 10)


def test_grow_residual_branch_updates_skip_and_fork_inputs():
    """
    Growing r4_b should align merge skip width and r4_a in_features at the fork.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.complex_residual_many_widths())
    x = torch.randn(2, 4)

    # Act
    AddNeurons(["r4_b", 1.5]).execute(TracedModel.create(gm, (1, 3, 32, 32)))
    y = gm(x)

    # Assert
    assert gm.r4_b.out_features == 16
    assert gm.merge.out_features == 16
    assert gm.r4_a.in_features == gm.merge.out_features
    assert gm.head.in_features == gm.r4_b.out_features
    assert y.shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grow_hidden_on_cuda_cifar_style_net():
    """
    Growing hidden on a CUDA GraphModule must keep all params on GPU and forward must work.
    """

    class TinyCifar(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.hidden = nn.Linear(64, 512)
            self.output = nn.Linear(512, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.flatten(self.pool(x))
            return self.output(self.hidden(x))

    # Arrange
    gm = fx.symbolic_trace(TinyCifar()).cuda()
    x = torch.randn(2, 3, 32, 32, device="cuda")

    # Act
    AddNeurons(["hidden", 1.5]).execute(TracedModel.create(gm, (1, 3, 32, 32)))
    y = gm(x)

    # Assert
    assert gm.hidden.out_features == 768
    assert gm.output.in_features == 768
    assert next(gm.hidden.parameters()).is_cuda
    assert next(gm.output.parameters()).is_cuda
    assert y.shape == (2, 10)
