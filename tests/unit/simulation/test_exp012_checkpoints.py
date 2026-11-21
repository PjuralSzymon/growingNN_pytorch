from types import SimpleNamespace

import torch

from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler
from experiments.train_cifar10_exp012_family_search import (
    save_decision_checkpoint, source_fingerprint, write_source_snapshot,
)
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel


def test_decision_checkpoint_restores_residual_graph(tmp_path):
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)[0].execute(traced)
    next(a for a in AddResConvLayer.generate_all_actions(traced)
         if a.params[:2] == ['conv1', 'seq_conv_0']).execute(traced)
    traced.gm.eval()
    scheduler = CosineDecayScheduler(total_epochs=200)
    scheduler.global_epoch = 40
    config = RunningConfig(generations=1, epochs=1, lr_scheduler=scheduler)
    config.experiment_board = SimpleNamespace(root=tmp_path, _current_generation=2)
    inputs = torch.randn(2, 3, 32, 32)
    before = traced.gm(inputs).detach()
    save_decision_checkpoint(traced, config)
    payload = torch.load(tmp_path / 'decision_checkpoints/generation_2.pt',
                         map_location='cpu', weights_only=False)
    assert payload['generation'] == 2
    assert payload['input_shape'] == (1, 3, 32, 32)
    assert payload['lr_scheduler'].global_epoch == 40
    torch.testing.assert_close(payload['model'](inputs), before)


def test_source_snapshot_matches_recorded_fingerprint(tmp_path):
    target = tmp_path / 'source_snapshot.zip'
    write_source_snapshot(target, source_fingerprint())
    assert target.is_file()
