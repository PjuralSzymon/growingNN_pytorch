from types import SimpleNamespace

from experiments.train_cifar10_exp014_action_replay import Phase3Board
from growingnn.core.config import RunningConfig
from growingnn.simulation.simulation_algorithms import montecarlo_alg_2 as search


class Model:
    def __init__(self):
        self.gm = []

    def param_count(self):
        return 1


class Action:
    def __init__(self, name):
        self.name = name

    def execute(self, model):
        model.gm.append(self.name)

    def __str__(self):
        return self.name


def test_search_reports_candidates_to_enabled_collector(monkeypatch):
    board = Phase3Board()
    config = RunningConfig(generations=1, epochs=1, enable_experiment_board=True,
                           experiment_board=board)
    config.simulation_scheduler = SimpleNamespace(simulation_time=0)
    config.simulation_score = SimpleNamespace(score=lambda model, cfg: 0.8 if model == ['b'] else 0.2)
    actions = [Action('a'), Action('b')]
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: actions)
    monkeypatch.setattr(search, 'clear_reshepers_cache', lambda: None)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr(search.project_config, 'SIMULATION_MIN_ALGORITHM_ITERATION_RUNS', 1)

    selected, _, _ = search.get_action(Model(), config)

    assert str(selected) == 'b'
    assert board.result is not None
    assert {row['action'] for row in board.result['candidates']} == {'a', 'b'}
    assert next(row for row in board.result['candidates'] if row['action'] == 'b')['chosen']
