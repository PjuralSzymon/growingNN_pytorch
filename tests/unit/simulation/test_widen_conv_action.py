import torch
import pytest

from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.widen_conv import WidenConv
from growingnn.core.traced_model import TracedModel
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_algorithms import montecarlo_alg_2
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
    monkeypatch.setattr(montecarlo_alg_2, 'generate_all_actions',
                        lambda model, cfg: [action] if model is traced else [])
    monkeypatch.setattr('growingnn.core.config.MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr('growingnn.core.config.MCTS_MEASUREMENTS_BEFORE_EXPAND', 1)
    chosen, depth, rollouts = montecarlo_alg_2.get_action(traced, config)
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


def _starter_with_residual_after_conv1():
    from growingnn.actions.add_res_conv_layer import AddResConvLayer

    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)[0].execute(traced)
    next(a for a in AddResConvLayer.generate_all_actions(traced)
         if a.params[:2] == ['conv1', 'seq_conv_0']).execute(traced)
    return traced


def test_widening_is_offered_after_a_residual_branch_and_preserves_outputs():
    import copy

    traced = _starter_with_residual_after_conv1()
    traced.gm.train()
    with torch.no_grad():
        for _ in range(3):
            traced.gm(torch.randn(8, 3, 32, 32) + 1)
    traced.gm.eval()
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16])
    twin = copy.deepcopy(traced)
    inputs = torch.randn(8, 3, 32, 32)
    before = traced.gm(inputs).detach()
    action.execute(traced)
    torch.rand(50)
    action.execute(twin)
    after = traced.gm(inputs)
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(after, twin.gm(inputs), rtol=0, atol=0)
    assert traced.gm.conv1.out_channels == 16
    assert traced.gm.seq_conv_0.in_channels == 16
    assert traced.gm.res_conv__0.in_channels == 16
    after.square().mean().backward()
    assert not torch.equal(traced.gm.conv1.weight.grad[:8], traced.gm.conv1.weight.grad[8:])


def test_widening_through_dropout_branch_preserves_eval_outputs():
    from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer

    traced = _starter_with_residual_after_conv1()
    next(a for a in AddSeqDropoutLayer.generate_all_actions(traced)
         if a.params[:2] == ['conv1', 'res_conv__0']).execute(traced)
    traced.gm.eval()
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['conv1', 16])
    inputs = torch.randn(4, 3, 32, 32)
    before = traced.gm(inputs).detach()
    action.execute(traced)
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-4, atol=1e-5)


def test_unsupported_consumer_is_not_offered():
    class Unsupported(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3)

        def forward(self, x):
            return self.conv(x).flatten(1)

    traced = TracedModel.create(Unsupported(), (1, 3, 32, 32))
    assert WidenConv.generate_all_actions(traced) == []
    assert traced.gm.conv.out_channels == 8


def _with_trained_batchnorm_statistics(traced):
    torch.manual_seed(0)
    with torch.no_grad():
        for module in traced.gm.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_()
                module.running_var.uniform_(0.5, 2.0)
                module.weight.normal_(1.0, 0.3)
                module.bias.normal_(0.0, 0.3)
    return traced


def test_every_convolution_is_offered_including_those_feeding_batchnorm_and_residual_sums():
    traced = _starter_with_residual_after_conv1()

    offered = {tuple(action.params[:2]) for action in WidenConv.generate_all_actions(traced)}

    assert offered == {(name, width) for name in ('conv1', 'seq_conv_0', 'res_conv__0') for width in (16, 32)}


@pytest.mark.parametrize('mode', ['eval', 'train'])
@pytest.mark.parametrize('name', ['conv1', 'seq_conv_0', 'res_conv__0'])
def test_widening_keeps_outputs_exactly_through_batchnorm_residual_sum_and_classifier(name, mode):
    import copy

    traced = _with_trained_batchnorm_statistics(_starter_with_residual_after_conv1())
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == [name, 16])
    reference = copy.deepcopy(traced.gm)
    getattr(reference, mode)()
    inputs = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        before = reference(inputs)
    parameters = sum(p.numel() for p in traced.gm.parameters())

    action.execute(traced)
    getattr(traced.gm, mode)()
    with torch.no_grad():
        after = traced.gm(inputs)

    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)
    assert sum(p.numel() for p in traced.gm.parameters()) > parameters


def test_widening_the_last_residual_group_widens_the_classifier_input_and_the_new_weights_learn():
    traced = _starter_with_residual_after_conv1()
    action = next(a for a in WidenConv.generate_all_actions(traced) if a.params[:2] == ['seq_conv_0', 16])

    action.execute(traced)

    assert traced.gm.seq_conv_0.out_channels == traced.gm.res_conv__0.out_channels == 16
    assert traced.gm.seq_conv_0_bn.num_features == 16
    assert traced.gm.linear2.in_features == 16
    assert torch.count_nonzero(traced.gm.linear2.weight[:, 8:]) == 0
    traced.gm(torch.randn(4, 3, 32, 32)).square().mean().backward()
    assert torch.count_nonzero(traced.gm.linear2.weight.grad[:, 8:]) > 0


def test_convolution_before_a_windowed_pool_and_flatten_is_not_offered():
    class PooledFlatten(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.head = torch.nn.Linear(8 * 16 * 16, 10)

        def forward(self, x):
            return self.head(torch.nn.functional.max_pool2d(self.conv(x), 2).flatten(1))

    traced = TracedModel.create(PooledFlatten(), (1, 3, 32, 32))

    assert WidenConv.generate_all_actions(traced) == []


def test_width_of_a_functional_pool_is_its_input_channel_count():
    from growingnn.utils.fx import NodeWidthAnalyser

    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    pool = next(n for n in traced.gm.graph.nodes if n.name == 'adaptive_avg_pool2d')

    assert NodeWidthAnalyser.node_output_width(traced.gm, pool) == 8


class SpatialFlattenNet(torch.nn.Module):
    """Flatten after pooling to 8x8, so the classifier reads channels x 64 features."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 8 * 8, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(torch.nn.functional.adaptive_avg_pool2d(x, (8, 8)).flatten(1))


def test_widening_before_a_spatial_flatten_leaves_the_classifier_alone():
    traced = TracedModel.create(SpatialFlattenNet(), (1, 1, 8, 8))
    AddSeqConvLayer.generate_all_actions(traced, add_batchnorm_and_relu=False, before_global_pool=True)[0].execute(traced)
    inputs = torch.randn(2, 1, 8, 8)
    before = traced.gm(inputs).detach()
    WidenConv(['conv', 8]).execute(traced)
    assert traced.gm.conv.out_channels == 8
    assert traced.gm.seq_conv_0.in_channels == 8
    assert traced.gm.fc.in_features == 4 * 8 * 8
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-5, atol=1e-6)
    assert all(action.params[0] != 'seq_conv_0' for action in WidenConv.generate_all_actions(traced))
