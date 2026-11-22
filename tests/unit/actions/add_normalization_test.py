import pytest
import torch
from torch import nn
import torch.nn.functional as F

from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.add_normalization import AddNormalization
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.registry import generate_all_actions
from growingnn.actions.widen_conv import WidenConv
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_editor import ModelStructureEditor


class Conv1dNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv1d(2, 6, 3, padding=1)
        self.second = nn.Conv1d(6, 6, 3, padding=1)
        self.head = nn.Linear(6, 4)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = F.relu(self.second(x))
        return self.head(x.mean(-1))


class Conv3dNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv3d(1, 4, 3, padding=1)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        x = F.relu(self.first(x))
        return self.head(F.adaptive_avg_pool3d(x, 1).flatten(1))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(5, 7)
        self.second = nn.Linear(7, 7)
        self.head = nn.Linear(7, 3)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = F.relu(self.second(x))
        return self.head(x)


class SequenceMlp(nn.Module):
    """A (batch, tokens, features) model, as in a transformer feed-forward block."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(5, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        return self.head(F.relu(self.first(x)).mean(1))


class StemWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(8)
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


CASES = [
    (Conv1dNet, (4, 2, 16), {'first': ('BatchNorm1d', 6), 'second': ('BatchNorm1d', 6)}),
    (Conv3dNet, (4, 1, 6, 6, 6), {'first': ('BatchNorm3d', 4)}),
    (Mlp, (4, 5), {'first': ('BatchNorm1d', 7), 'second': ('BatchNorm1d', 7)}),
    (SequenceMlp, (4, 6, 5), {'first': ('LayerNorm', 8)}),
    (TinyCifar10Starter, (4, 3, 32, 32), {'conv1': ('BatchNorm2d', 8)}),
]


def _offers(traced):
    return {action.params[0]: tuple(action.params[1:3]) for action in AddNormalization.generate_all_actions(traced)}


def _apply(model_type, shape, layer):
    torch.manual_seed(0)
    model = model_type()
    traced = TracedModel.create(model, (1, *shape[1:]))
    action = next(a for a in AddNormalization.generate_all_actions(traced) if a.params[0] == layer)
    return traced, action


@pytest.mark.parametrize('model_type, shape, expected', CASES, ids=lambda value: getattr(value, '__name__', None))
def test_offers_the_normalization_that_fits_each_layer_and_never_the_output_layer(model_type, shape, expected):
    traced = TracedModel.create(model_type(), (1, *shape[1:]))
    assert _offers(traced) == expected


@pytest.mark.parametrize('model_type, shape, expected', CASES, ids=lambda value: getattr(value, '__name__', None))
def test_new_normalization_keeps_the_evaluation_output_and_learns(model_type, shape, expected):
    for layer in expected:
        traced, action = _apply(model_type, shape, layer)
        inputs = torch.randn(*shape)
        traced.gm.eval()
        before = traced.gm(inputs).detach()
        parameters = traced.param_count()
        action.execute(traced)
        traced.gm.eval()
        if action.params[1] == 'LayerNorm':
            # LayerNorm normalizes each sample in both modes, so its default init changes the output.
            assert not torch.allclose(traced.gm(inputs), before, atol=1e-4)
        else:
            torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-4, atol=1e-5)
        norm = getattr(traced.gm, action.params[3])
        assert type(norm).__name__ == action.params[1]
        assert traced.param_count() == parameters + 2 * action.params[2]
        traced.gm.train()
        traced.gm(inputs).square().mean().backward()
        assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0


def test_training_mode_output_changes_because_batch_statistics_normalize():
    traced, action = _apply(TinyCifar10Starter, (8, 3, 32, 32), 'conv1')
    inputs = torch.randn(8, 3, 32, 32)
    traced.gm.train()
    before = traced.gm(inputs).detach()
    action.execute(traced)
    traced.gm.train()
    assert not torch.allclose(traced.gm(inputs), before, atol=1e-4)


def test_no_second_normalization_after_a_normalized_layer():
    traced = TracedModel.create(StemWithBatchNorm(), (1, 3, 32, 32))
    assert _offers(traced) == {}
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    AddNormalization.generate_all_actions(traced)[0].execute(traced)
    assert _offers(traced) == {}


def test_normalization_follows_the_residual_sum_into_a_layer():
    traced = TracedModel.create(Mlp(), (1, 5))
    ModelStructureEditor.add_new_residual_layer(traced.gm, 'first', 'second', nn.Linear(7, 7), 'skip')
    traced.invalidate()
    action = next(a for a in AddNormalization.generate_all_actions(traced) if a.params[0] == 'second')
    action.execute(traced)
    norm_node = next(node for node in traced.gm.graph.nodes if node.target == action.params[3])
    assert norm_node.args[0].op == 'call_function'
    assert 'skip' not in _offers(traced)


def test_starter_reaches_the_residual_starter_stem():
    """Widening the Experiment 010 starter to 16 channels and normalizing it gives conv -> BatchNorm -> ReLU."""
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16]).execute(traced)
    AddNormalization.generate_all_actions(traced)[0].execute(traced)
    nodes = [node for node in traced.gm.graph.nodes if node.op in ('call_module', 'call_function')]
    conv = next(node for node in nodes if node.target == 'conv1')
    norm = list(conv.users)[0]
    activation = list(norm.users)[0]
    assert isinstance(getattr(traced.gm, norm.target), nn.BatchNorm2d)
    assert getattr(traced.gm, norm.target).num_features == 16
    assert 'relu' in str(activation.target)
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_widening_after_normalization_grows_the_batch_norm():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    action = AddNormalization.generate_all_actions(traced)[0]
    action.execute(traced)
    widen = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16])
    widen.execute(traced)
    assert getattr(traced.gm, action.params[3]).num_features == 16
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_registry_offers_normalization_and_widening_by_default():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    config = RunningConfig(generations=1, epochs=1)
    actions = generate_all_actions(traced, config)
    assert any(isinstance(a, AddNormalization) for a in actions)
    assert any(isinstance(a, WidenConv) for a in actions)
    config.ACTIONS_ENABLE_ADD_NORMALIZATION = False
    assert not any(isinstance(a, AddNormalization) for a in generate_all_actions(traced, config))
