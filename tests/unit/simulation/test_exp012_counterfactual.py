import torch
from torch import nn

from experiments.evaluate_exp012_checkpoint import evaluate_candidates, heldout_validation_loader
from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler


def test_counterfactual_scores_unchanged_and_action_without_mutating_source():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    action = AddSeqConvLayer.generate_all_actions(
        traced, add_batchnorm_and_relu=True, before_global_pool=True)[0]
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    cfg = RunningConfig(generations=1, epochs=1, device='cpu', criterion=nn.CrossEntropyLoss(),
                        simulation_score=SimulationScore(weight_acc=1, weight_countW=0),
                        simulation_scheduler=AlwaysSimulationScheduler(1, 1))
    cfg.set_simulation_loaders(loader, loader)
    before = {name: value.clone() for name, value in traced.gm.state_dict().items()}
    rows = evaluate_candidates(traced, [action], cfg, loader, loader, seed=10, epochs=1)
    assert [row['action'] == 'unchanged' for row in rows] == [True, False]
    assert rows[0]['validation_gain_vs_unchanged'] == 0
    assert all(0 <= row['simulation_score'] <= 1 for row in rows)
    assert rows[1]['parameter_count'] > rows[0]['parameter_count']
    for name, value in traced.gm.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_counterfactual_validation_excludes_search_examples():
    dataset = torch.utils.data.TensorDataset(torch.arange(10), torch.arange(10))
    val = torch.utils.data.DataLoader(dataset, batch_size=2)
    sim = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [1, 3, 5]), batch_size=2)
    heldout = heldout_validation_loader(val, sim)
    assert set(heldout.dataset.indices) == {0, 2, 4, 6, 7, 8, 9}
