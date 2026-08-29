"""Unit tests for ``growingnn.actions.add_neurons`` neuron-grow propagation."""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_neurons import AddNeurons, expand_layer_output
from growingnn.core import config
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


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


def test_add_neurons_keeps_linear_in_features_after_spatial_flatten():
    """
    Growing hidden linear after conv MaxPool flatten should keep in_features at C*H*W.
    """
    # Arrange
    class CifarStem(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 4
            spatial = 16
            self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.linear = nn.Linear(channels * spatial * spatial, 32)
            self.linear2 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = F.relu(self.linear(torch.flatten(x, 1)))
            return self.linear2(x)

    gm = fx.symbolic_trace(CifarStem())
    x = torch.randn(2, 3, 32, 32)

    # Act
    AddNeurons(["linear", 1.1]).execute(TracedModel.create(gm, (1, 3, 32, 32)))
    y = gm(x)

    # Assert
    assert gm.linear.in_features == 1024
    assert gm.linear.out_features == 35
    assert gm.linear2.in_features == 35
    assert y.shape == (2, 10)


def test_add_neurons_spatial_flatten_does_not_print_shapeprop_traceback(capsys):
    """
    Growing a CIFAR-style hidden linear should not dump ShapeProp tracebacks to stderr.
    """
    # Arrange
    class CifarStem(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 4
            spatial = 16
            self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.linear = nn.Linear(channels * spatial * spatial, 32)
            self.linear2 = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = F.relu(self.linear(torch.flatten(x, 1)))
            return self.linear2(x)

    gm = fx.symbolic_trace(CifarStem())

    # Act
    AddNeurons(["linear", 1.1]).execute(TracedModel.create(gm, (1, 3, 32, 32)))
    err = capsys.readouterr().err

    # Assert
    assert "ShapeProp error" not in err
    assert "mat1 and mat2 shapes cannot be multiplied" not in err
