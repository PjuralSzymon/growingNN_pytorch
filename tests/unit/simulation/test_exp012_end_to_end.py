import torch
from torch import nn

from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from experiments.train_cifar10_exp011_budgeted_mcts import IsolatedSearch
from experiments.train_cifar10_exp012_family_search import FamilySearch
from growingnn.board import ExperimentBoard
from growingnn.core.config import RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.training.trainer import train_generations


def test_family_experiment_trains_searches_and_saves_checkpoint(tmp_path, monkeypatch):
    board = ExperimentBoard(tmp_path)
    monkeypatch.setattr(board, 'save_graphs', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(board, 'save_candidate_graph', lambda *_args: None)
    monkeypatch.setattr('growingnn.core.config.MCTS_ROLLOUT_DEPTH', 0)
    data = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    config = RunningConfig(generations=2, epochs=1, device='cpu', quiet=True,
                           criterion=nn.CrossEntropyLoss(),
                           simulation_alg=IsolatedSearch(FamilySearch()),
                           simulation_score=SimulationScore(weight_countW=0),
                           simulation_scheduler=AlwaysSimulationScheduler(0, 1),
                           enable_experiment_board=True, experiment_board=board)
    trained, history = train_generations(TinyCifar10Starter(), loader, loader, config,
                                         sim_train_loader=loader, sim_val_loader=loader)
    assert len(history['generation']) == 2
    assert len(history['action_name']) == 1
    assert (tmp_path / 'decision_checkpoints/generation_0.pt').exists()
    checkpoint = torch.load(tmp_path / 'decision_checkpoints/generation_0.pt',
                            map_location='cpu', weights_only=False)
    assert checkpoint['model'](torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert trained(torch.randn(2, 3, 32, 32)).shape == (2, 10)
