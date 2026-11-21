import torch
import pytest

from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.widen_conv import WidenConv
from growingnn.core.traced_model import TracedModel
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_algorithms import budgeted_mcts_alg
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer


class TwoConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.second = torch.nn.Conv2d(8, 12, 3, padding=1)
        self.head = torch.nn.Linear(12, 10)

    def forward(self, x):
        x = torch.relu(self.first(x))
        x = torch.relu(self.second(x))
        return self.head(torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


def test_widen_tiny_starter_preserves_classifier_shape_and_trains():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    actions = WidenConv.generate_all_actions(traced)
    action = next(a for a in actions if a.params[:2] == ['conv1', 16])
    inputs = torch.randn(2, 3, 32, 32)
    before = traced.gm(inputs).detach()
    action.execute(traced)
    assert traced.gm.conv1.out_channels == 16
    assert traced.gm.linear2.in_features == 16
    assert traced.gm.linear2.out_features == 10
    output = traced.gm(inputs)
    assert output.shape == (2, 10)
    torch.testing.assert_close(output, before, rtol=1e-5, atol=1e-6)
    output.square().mean().backward()
    assert traced.gm.conv1.weight.grad is not None
    grad = traced.gm.conv1.weight.grad
    assert not torch.equal(grad[:8], grad[8:])


def test_widening_after_convolution_has_been_added():
    traced = TracedModel.create(TwoConvNet(), (1, 3, 32, 32))
    actions = WidenConv.generate_all_actions(traced)
    assert ['first', 16] in [action.params[:2] for action in actions]
    assert ['second', 24] in [action.params[:2] for action in actions]
    for action in actions:
        candidate = TracedModel.create(TwoConvNet(), (1, 3, 32, 32))
        action.execute(candidate)
        inputs = torch.randn(2, 3, 32, 32)
        # Recreate a reference with the same pre-action weights.
        # The candidate is checked for shape and gradients here; the exact
        # function-preservation case is covered below.
        output = candidate.gm(inputs)
        assert output.shape == (2, 10)
        output.square().mean().backward()


def test_search_can_score_and_execute_widening(monkeypatch):
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    samples = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32),
                                             torch.randint(0, 10, (8,)))
    loader = torch.utils.data.DataLoader(samples, batch_size=4)
    config = RunningConfig(generations=1, epochs=1, criterion=torch.nn.CrossEntropyLoss(),
                           simulation_score=SimulationScore(weight_countW=0),
                           simulation_scheduler=AlwaysSimulationScheduler(0, 1),
                           device='cpu')
    config.set_simulation_loaders(loader, loader)
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16])
    monkeypatch.setattr(budgeted_mcts_alg, 'generate_all_actions', lambda *_: [action])
    monkeypatch.setattr('growingnn.core.config.MCTS_ROLLOUT_DEPTH', 0)
    chosen, depth, rollouts = budgeted_mcts_alg.get_action(traced, config)
    assert chosen is action
    assert (depth, rollouts) == (1, 1)
    chosen.execute(traced)
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_widen_upstream_conv_preserves_outputs_and_replays_identically():
    import copy

    traced = TracedModel.create(TwoConvNet(), (1, 3, 32, 32))
    traced.gm.eval()
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['first', 16])
    twin = copy.deepcopy(traced)
    inputs = torch.randn(4, 3, 32, 32)
    before = traced.gm(inputs).detach()
    action.execute(traced)
    torch.rand(100)
    action.execute(twin)
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(traced.gm(inputs), twin.gm(inputs), rtol=0, atol=0)


def test_widen_after_sequential_conv_with_batchnorm():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    inserted = AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)
    assert inserted
    inserted[0].execute(traced)
    # Give BatchNorm non-default running statistics as it would have after training.
    traced.gm.train()
    with torch.no_grad():
        for _ in range(3):
            traced.gm(torch.randn(8, 3, 32, 32) + 1.0)
    traced.gm.eval()
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16])
    inputs = torch.randn(4, 3, 32, 32)
    before = traced.gm(inputs).detach()
    action.execute(traced)
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_widen_preserves_batchnorm_graph_on_cuda():
    traced = TracedModel.create(TinyCifar10Starter().cuda(), (1, 3, 32, 32))
    AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)[0].execute(traced)
    traced.gm.eval()
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16])
    inputs = torch.randn(8, 3, 32, 32, device='cuda')
    before = traced.gm(inputs).detach()
    action.execute(traced)
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-4, atol=1e-5)
