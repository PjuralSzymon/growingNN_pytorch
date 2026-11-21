from types import SimpleNamespace
from unittest.mock import Mock

from growingnn.simulation.simulation_algorithms import montecarlo_alg_2 as search


class Model:
    def __init__(self):
        self.gm = []


class Action:
    def __init__(self, name):
        self.name = name

    def execute(self, model):
        model.gm.append(self.name)


def test_expand_creates_children_without_scoring(monkeypatch):
    scoring = Mock()
    config = SimpleNamespace(simulation_score=SimpleNamespace(score=scoring))
    actions = [Action('a'), Action('b')]
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: actions)
    node = search.TreeNode(None, None, Model(), config)

    node.expand()

    assert [child.traced.gm for child in node.child_nodes] == [['a'], ['b']]
    assert node.traced.gm == []
    scoring.assert_not_called()


def test_rollout_scores_each_step_and_weights_nearer_states(monkeypatch):
    scoring = Mock(side_effect=[0.4, 0.8, 0.2])
    config = SimpleNamespace(experiment_board=None, simulation_score=SimpleNamespace(score=scoring))
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action('next')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 2)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_WEIGHT_DECAY', 0.5)
    node = search.TreeNode(None, None, Model(), config)

    assert abs(node.rollout() - (0.4 + 0.5 * 0.8 + 0.25 * 0.2) / 1.75) < 1e-9

    assert [call.args[0] for call in scoring.call_args_list] == [[], ['next'], ['next', 'next']]
    assert node.traced.gm == []


def test_rollout_reuses_initial_score_and_uses_configured_decay(monkeypatch):
    scoring = Mock(side_effect=[0.4, 0.8, 0.6])
    config = SimpleNamespace(experiment_board=None, simulation_score=SimpleNamespace(score=scoring))
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action('next')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 1)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_WEIGHT_DECAY', 0.25)
    node = search.TreeNode(None, None, Model(), config)
    assert abs(node.rollout() - (0.4 + 0.25 * 0.8) / 1.25) < 1e-9
    assert abs(node.rollout() - (0.4 + 0.25 * 0.6) / 1.25) < 1e-9
    assert scoring.call_count == 3


def test_final_choice_uses_mean_not_exploration_bonus(monkeypatch):
    monkeypatch.setattr(search.project_config, 'MCTS_UCB1_USE_SQRT', True)
    monkeypatch.setattr(search.project_config, 'MCTS_UCB1_C', 1.0)
    root = search.TreeNode(None, None, Model(), None)
    strong = search.TreeNode(root, Action('strong'), Model(), None)
    uncertain = search.TreeNode(root, Action('uncertain'), Model(), None)
    unvisited = search.TreeNode(root, Action('unvisited'), Model(), None)
    strong.visit_counter, strong.value = 10, 8.0
    uncertain.visit_counter, uncertain.value = 1, 0.5
    root.visit_counter = 11
    root.child_nodes = [strong, uncertain]
    assert root.get_best_child() is uncertain
    root.child_nodes.append(unvisited)
    assert root.get_final_child() is strong
    root.child_nodes = [unvisited]
    assert root.get_final_child() is None
