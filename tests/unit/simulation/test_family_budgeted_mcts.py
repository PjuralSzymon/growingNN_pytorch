import copy
import json
from types import SimpleNamespace

import torch
from unittest.mock import patch

from experiments import family_budgeted_mcts as search
from experiments.family_budgeted_mcts import FanoutWidenConv, _fanout_candidates
from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.traced_model import TracedModel
from growingnn.core.config import RunningConfig
from growingnn.board import ExperimentBoard
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler


def test_fanout_candidates_include_safe_widening_on_replayed_graph():
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    AddSeqConvLayer.generate_all_actions(traced, add_batchnorm_and_relu=True,
                                        before_global_pool=True)[0].execute(traced)
    next(a for a in AddResConvLayer.generate_all_actions(traced)
         if a.params[:2] == ['conv1', 'seq_conv_0']).execute(traced)
    traced.gm.eval()
    candidates = _fanout_candidates(traced, [])
    action = next(a for a in candidates if a.params[:2] == ['conv1', 16])
    assert isinstance(action, FanoutWidenConv)
    candidate = copy.deepcopy(traced)
    inputs = torch.randn(3, 3, 32, 32)
    before = traced.gm(inputs).detach()
    action.execute(candidate)
    torch.testing.assert_close(candidate.gm(inputs), before, rtol=1e-4, atol=1e-5)


def test_search_grades_different_families_before_repeating_one():
    class A:
        def execute(self, model):
            model.score = 0.1

    class B:
        def execute(self, model):
            model.score = 0.8

    class C:
        def execute(self, model):
            model.score = 0.4

    actions = [A(), A(), A(), B(), C()]
    config = SimpleNamespace(experiment_board=None,
                             simulation_scheduler=SimpleNamespace(simulation_time=1))
    with (patch.object(search, 'generate_all_actions', return_value=actions),
          patch.object(search, '_fanout_candidates', return_value=[]),
          patch.object(search.TreeNode, 'rollout', lambda node: node.traced.score),
          patch.object(search.time, 'monotonic', side_effect=[0, 0, 0, 2]),
          patch.object(search, 'clear_reshepers_cache')):
        chosen, depth, rollouts = search.get_action(SimpleNamespace(), config)
    assert isinstance(chosen, B)
    assert (depth, rollouts) == (1, 3)


def test_family_search_scores_real_candidate(monkeypatch):
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    config = RunningConfig(generations=1, epochs=1, device='cpu',
                           criterion=torch.nn.CrossEntropyLoss(),
                           simulation_score=SimulationScore(weight_countW=0),
                           simulation_scheduler=AlwaysSimulationScheduler(0, 1))
    config.ACTIONS_ENABLE_WIDEN_CONV = True
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    config.set_simulation_loaders(loader, loader)
    monkeypatch.setattr('growingnn.core.config.MCTS_ROLLOUT_DEPTH', 0)
    action, depth, rollouts = search.get_action(traced, config)
    assert action is not None
    assert (depth, rollouts) == (1, 1)
    action.execute(traced)
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_family_search_exports_actual_recommendation(tmp_path, monkeypatch):
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    board = ExperimentBoard(tmp_path)
    config = RunningConfig(generations=1, epochs=1, device='cpu',
                           criterion=torch.nn.CrossEntropyLoss(),
                           simulation_score=SimulationScore(weight_countW=0),
                           simulation_scheduler=AlwaysSimulationScheduler(0, 1),
                           enable_experiment_board=True, experiment_board=board)
    config.ACTIONS_ENABLE_WIDEN_CONV = True
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    config.set_simulation_loaders(loader, loader)
    monkeypatch.setattr('growingnn.core.config.MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr(board, 'save_candidate_graph', lambda *_args: None)
    board.bind_config(config)
    action, _, _ = search.get_action(traced, config)
    record = json.loads((tmp_path / 'simulations/simulation_gen_0.json').read_text())
    chosen = [row for row in record['candidates'] if row['chosen']]
    assert len(chosen) == 1
    assert chosen[0]['action'] == str(action) == record['actionChosen']
    assert record['scoreChosen'] == chosen[0]['score']
