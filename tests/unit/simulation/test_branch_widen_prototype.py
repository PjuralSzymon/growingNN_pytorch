import copy

import pytest
import torch
from torch import nn

from experiments.branch_widen_prototype import widen_fanout
from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.traced_model import TracedModel


def test_widen_preserves_the_replayed_residual_graph_and_replays_seed():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)[0].execute(traced)
    residual = next(a for a in AddResConvLayer.generate_all_actions(traced)
                    if a.params[:2] == ['conv1', 'seq_conv_0'])
    residual.execute(traced)
    traced.gm.train()
    with torch.no_grad():
        for _ in range(3):
            traced.gm(torch.randn(8, 3, 32, 32) + 1)
    traced.gm.eval()
    twin = copy.deepcopy(traced.gm)
    inputs = torch.randn(8, 3, 32, 32)
    before = traced.gm(inputs).detach()
    widen_fanout(traced.gm, 'conv1', 16, 42)
    torch.rand(50)
    widen_fanout(twin, 'conv1', 16, 42)
    after = traced.gm(inputs)
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(after, twin(inputs), rtol=0, atol=0)
    assert traced.gm.conv1.out_channels == 16
    assert traced.gm.seq_conv_0.in_channels == 16
    assert traced.gm.res_conv__0.in_channels == 16
    after.square().mean().backward()
    assert not torch.equal(traced.gm.conv1.weight.grad[:8], traced.gm.conv1.weight.grad[8:])


def test_unsupported_branch_rejected_before_mutation():
    class Unsupported(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)

        def forward(self, x):
            return self.conv(x).flatten(1)

    traced = TracedModel.create(Unsupported(), (1, 3, 32, 32))
    with pytest.raises(ValueError):
        widen_fanout(traced.gm, 'conv', 16, 1)
    assert traced.gm.conv.out_channels == 8


def test_widen_through_dropout_branch_preserves_eval_outputs():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)[0].execute(traced)
    next(a for a in AddResConvLayer.generate_all_actions(traced)
         if a.params[:2] == ['conv1', 'seq_conv_0']).execute(traced)
    next(a for a in AddSeqDropoutLayer.generate_all_actions(traced)
         if a.params[:2] == ['conv1', 'res_conv__0']).execute(traced)
    traced.gm.eval()
    inputs = torch.randn(4, 3, 32, 32)
    before = traced.gm(inputs).detach()
    widen_fanout(traced.gm, 'conv1', 16, 42)
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-4, atol=1e-5)
