"""Small CPU smoke comparison of root-action coverage, not an accuracy test.

Replays the first two v3 architecture edits on fresh weights, uses small
disjoint subsets of CIFAR-10 training images, and reports sampled families.
The current source is used; it is not a replay of the original training run.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from experiments import family_budgeted_mcts
from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.registry import generate_all_actions
from growingnn.board import ExperimentBoard
from growingnn.core import config as project_config
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_algorithms import budgeted_mcts_alg
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.utils.seed import seed_all


def main():
    torch.set_num_threads(2)
    seed_all(1)
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    config = RunningConfig(generations=1, epochs=1, device='cpu', criterion=nn.CrossEntropyLoss(),
                           simulation_score=SimulationScore(weight_countW=0),
                           simulation_scheduler=AlwaysSimulationScheduler(5, 1))
    config.ACTIONS_ENABLE_WIDEN_CONV = True
    config.MCTS_MAX_TREE_DEPTH = 1
    board_path = ROOT / 'experiments/output/train_cifar10/runs/exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/board'
    for index in range(2):
        sim = json.loads((board_path / f'simulations/simulation_gen_{index}.json').read_text())
        action = next(a for a in generate_all_actions(traced, config) if str(a) == sim['actionChosen'])
        action.execute(traced)
    data = CIFAR10(str(ROOT / 'experiments/data/cifar10'), train=True, download=False, transform=ToTensor())
    train = DataLoader(Subset(data, range(64)), batch_size=16, shuffle=True)
    val = DataLoader(Subset(data, range(64, 96)), batch_size=16)
    config.set_simulation_loaders(train, val)
    project_config.MCTS_ROLLOUT_DEPTH = 0
    project_config.MCTS_UCB1_USE_SQRT = True
    with tempfile.TemporaryDirectory() as directory:
        for name, method in [('budgeted', budgeted_mcts_alg), ('family', family_budgeted_mcts)]:
            seed_all(123)
            board = ExperimentBoard(Path(directory) / name)
            board.save_candidate_graph = lambda *_args: None
            config.experiment_board = board
            board.bind_config(config)
            action, _, count = method.get_action(copy.deepcopy(traced), config)
            sim = json.loads((board.root / 'simulations/simulation_gen_0.json').read_text())
            families = sorted({candidate['name'] for candidate in sim['candidates']})
            print(f'{name}: {count} evaluations, {len(families)} displayed families, '
                  f'{sim["durationSec"]:.2f}s, chosen={action}')
            print('  ' + ', '.join(families))


if __name__ == '__main__':
    main()
