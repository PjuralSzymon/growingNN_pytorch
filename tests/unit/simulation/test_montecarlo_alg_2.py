import copy
from types import SimpleNamespace
from unittest.mock import Mock

from growingnn.simulation.simulation_algorithms import montecarlo_alg_2 as search


class FakeScore:
    """Stands in for SimulationScore; the 'trained' model is an unchanged copy."""

    def __init__(self, score):
        self.score = score

    def score_and_train(self, gm, cfg):
        return self.score(gm, cfg), copy.deepcopy(gm)


class Model:
    def __init__(self):
        self.gm = []
        self.input_shape = (1,)


class Action:
    def __init__(self, name):
        self.name = name

    def execute(self, model):
        model.gm.append(self.name)


def test_expand_builds_children_only_when_selected(monkeypatch):
    scoring = Mock()
    config = SimpleNamespace(simulation_score=FakeScore(scoring))
    actions = [Action('a'), Action('b')]
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: actions)
    node = search.TreeNode(None, None, Model(), config)

    node.expand()
    assert node.child_nodes == [] and not node.is_leaf()
    assert node.get_best_child() is None  # UCB alone never builds a child

    first = node.select_child()
    assert len(node.child_nodes) == 1
    second = node.select_child()

    assert sorted([first.traced.gm, second.traced.gm]) == [['a'], ['b']]
    assert node.traced.gm == []
    scoring.assert_not_called()


def test_node_without_actions_is_terminal_after_expansion(monkeypatch):
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [])
    node = search.TreeNode(None, None, Model(), SimpleNamespace())
    node.value, node.visit_counter = 0.4, 1

    value, _, rollouts = search._simulate(node)

    assert (value, rollouts) == (0.4, 0)
    assert not node.is_leaf() and node.child_nodes == []


def test_search_builds_only_the_children_it_visits(monkeypatch):
    from growingnn.core.config import RunningConfig

    built = []

    class Counted(Action):
        def execute(self, model):
            built.append(self.name)
            super().execute(model)

    config = RunningConfig(generations=1, epochs=1)
    config.simulation_scheduler = SimpleNamespace(simulation_time=0)
    config.simulation_score = FakeScore(lambda gm, cfg: 0.5)
    # Three root actions, each with twenty possible follow-ups.
    monkeypatch.setattr(search, 'generate_all_actions',
                        lambda model, cfg: [Counted(f'r{i}') for i in range(3)] if model.gm == []
                        else [Counted(f'f{i}') for i in range(20)])
    monkeypatch.setattr(search, 'clear_reshepers_cache', lambda: None)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr(search.project_config, 'SIMULATION_MIN_ALGORITHM_ITERATION_RUNS', 5)

    _, _, rollouts = search.get_action(Model(), config)

    # Each rollout scores one newly built child; the other follow-ups are never built.
    assert rollouts == 5
    assert len(built) == rollouts


def test_rollout_scores_each_step_and_weights_nearer_states(monkeypatch):
    scoring = Mock(side_effect=[0.4, 0.8, 0.2])
    config = SimpleNamespace(experiment_board=None, simulation_score=FakeScore(scoring))
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action('next')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 2)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_WEIGHT_DECAY', 0.5)
    node = search.TreeNode(None, None, Model(), config)

    assert abs(node.rollout() - (0.4 + 0.5 * 0.8 + 0.25 * 0.2) / 1.75) < 1e-9

    assert [call.args[0] for call in scoring.call_args_list] == [[], ['next'], ['next', 'next']]
    assert node.traced.gm == []


def test_rollout_reuses_initial_score_and_uses_configured_decay(monkeypatch):
    scoring = Mock(side_effect=[0.4, 0.8, 0.6])
    config = SimpleNamespace(experiment_board=None, simulation_score=FakeScore(scoring))
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


def test_rollout_remembers_only_its_own_score(monkeypatch):
    scoring = Mock(side_effect=[0.4, 0.8])
    config = SimpleNamespace(experiment_board=None, simulation_score=FakeScore(scoring))
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action('next')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 1)
    node = search.TreeNode(None, None, Model(), config)
    node.rollout()
    assert node._own_score == 0.4


def test_rollout_metrics_record_weighted_composite(monkeypatch):
    board = SimpleNamespace(simulation_metrics={'composite_score': 0.8, 'weight_acc_score': 0.8})
    config = SimpleNamespace(experiment_board=board,
                             simulation_score=FakeScore(Mock(side_effect=[0.4, 0.8])))
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action('next')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 1)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_WEIGHT_DECAY', 0.5)
    node = search.TreeNode(None, None, Model(), config)
    value = node.rollout()
    assert abs(value - (0.4 + 0.5 * 0.8) / 1.5) < 1e-9
    assert node.rollout_metrics['composite_score'] == value
    assert board.simulation_metrics['composite_score'] == 0.8


def test_legacy_ucb_exploits_mean_not_sum(monkeypatch):
    monkeypatch.setattr(search.project_config, 'MCTS_UCB1_USE_SQRT', False)
    monkeypatch.setattr(search.project_config, 'MCTS_UCB1_C', 0.0)
    root = search.TreeNode(None, None, Model(), None)
    many_weak = search.TreeNode(root, Action('weak'), Model(), None)
    few_strong = search.TreeNode(root, Action('strong'), Model(), None)
    many_weak.visit_counter, many_weak.value = 10, 3.0   # mean 0.3, sum 3.0
    few_strong.visit_counter, few_strong.value = 1, 0.9  # mean 0.9
    root.visit_counter = 11
    root.child_nodes = [many_weak, few_strong]
    assert root.get_best_child() is few_strong


def test_backprop_keeps_true_mean_values(monkeypatch):
    # Root value must be the plain sum of the rollout values each iteration returned,
    # for both settings of the (now ignored) propagation flag.
    scores = {(): 0.5, ('a',): 0.2, ('b',): 0.8, ('a', 'a'): 0.3, ('a', 'b'): 0.1,
              ('b', 'a'): 0.6, ('b', 'b'): 1.0}
    for flag in (False, True):
        monkeypatch.setattr(search.project_config, 'MCTS_PROPAGATE_ROLLOUT_VALUE', flag)
        monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
        monkeypatch.setattr(search.project_config, 'MCTS_UCB1_USE_SQRT', True)
        config = SimpleNamespace(experiment_board=None,
                                 simulation_score=FakeScore(lambda gm, cfg: scores[tuple(gm)]))
        monkeypatch.setattr(search, 'generate_all_actions',
                            lambda model, cfg: [Action('a'), Action('b')] if len(model.gm) < 2 else [])
        root = search.TreeNode(None, None, Model(), config)
        returned = [search._simulate(root)[0] for _ in range(8)]
        assert root.visit_counter == 8
        assert abs(root.value - sum(returned)) < 1e-9
        assert all(0.0 <= value <= 1.0 for value in returned)
        for child in root.child_nodes:
            assert 0.0 <= child.value / child.visit_counter <= 1.0


def test_chosen_row_matches_selected_action_when_board_sorts_rows(monkeypatch):
    from experiments.train_cifar10_exp014_action_replay import Phase3Board
    from growingnn.core.config import RunningConfig

    class SortingBoard(Phase3Board):
        # ExperimentBoard sorts rows by UCB score, so row order differs from child order.
        def mcts_candidates_from_root(self, root, config):
            return list(reversed(super().mcts_candidates_from_root(root, config)))

    class Named(Action):
        def __str__(self):
            return self.name

    board = SortingBoard()
    config = RunningConfig(generations=1, epochs=1, enable_experiment_board=True, experiment_board=board)
    config.simulation_scheduler = SimpleNamespace(simulation_time=0)
    config.simulation_score = FakeScore(lambda model, cfg: 0.9 if model == ['y'] else 0.1)
    actions = [Named('x'), Named('y')]
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: actions if model.gm == [] else [])
    monkeypatch.setattr(search, 'clear_reshepers_cache', lambda: None)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr(search.project_config, 'SIMULATION_MIN_ALGORITHM_ITERATION_RUNS', 1)
    model = Model()
    model.param_count = lambda: 1

    selected, _, _ = search.get_action(model, config)

    assert selected is actions[1]
    assert {row['action']: row['chosen'] for row in board.result['candidates']} == {
        'x': False, 'y': True}


def test_search_skips_root_rollout_and_covers_each_root_action_once(monkeypatch):
    from experiments.train_cifar10_exp014_action_replay import Phase3Board
    from growingnn.core.config import RunningConfig

    scored = []
    board = Phase3Board()
    config = RunningConfig(generations=1, epochs=1, enable_experiment_board=True, experiment_board=board)
    config.simulation_scheduler = SimpleNamespace(simulation_time=0)
    config.simulation_score = FakeScore(lambda gm, cfg: scored.append(list(gm)) or 0.5)
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action(x) for x in 'abc'])
    monkeypatch.setattr(search, 'clear_reshepers_cache', lambda: None)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr(search.project_config, 'SIMULATION_MIN_ALGORITHM_ITERATION_RUNS', 1)
    model = Model()
    model.param_count = lambda: 1

    _, _, rollouts = search.get_action(model, config)

    assert [] not in scored  # the starting model itself is never rolled out
    assert sorted(scored) == [['a'], ['b'], ['c']]
    assert rollouts == 3


class Conv(Action):
    pass


class Dropout(Action):
    pass


def test_expansion_order_cycles_through_action_types():
    actions = [Dropout(f'd{i}') for i in range(6)] + [Conv('c0'), Conv('c1')]
    for _ in range(20):
        order = search._type_interleaved(actions)
        assert sorted(a.name for a in order) == sorted(a.name for a in actions)
        # Both types appear in the first two and the first four positions.
        assert {type(a) for a in order[:2]} == {Conv, Dropout}
        assert sum(isinstance(a, Conv) for a in order[:4]) == 2


def test_random_follow_up_picks_action_type_uniformly(monkeypatch):
    import random
    random.seed(0)
    actions = [Dropout(f'd{i}') for i in range(9)] + [Conv('c0')]
    picks = [type(search._random_follow_up(actions)) for _ in range(2000)]
    assert 0.4 < picks.count(Conv) / len(picks) < 0.6


def test_every_scoring_run_in_a_search_uses_the_same_training_seed(monkeypatch):
    import torch
    from growingnn.core.config import RunningConfig

    draws = []

    def score(gm, cfg):
        draws.append(torch.rand(1).item())
        return 0.5

    config = RunningConfig(generations=1, epochs=1)
    config.simulation_scheduler = SimpleNamespace(simulation_time=0)
    config.simulation_score = FakeScore(score)
    monkeypatch.setattr(search, 'generate_all_actions',
                        lambda model, cfg: [Action('a'), Action('b'), Action('c')] if model.gm == [] else [])
    monkeypatch.setattr(search, 'clear_reshepers_cache', lambda: None)
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
    monkeypatch.setattr(search.project_config, 'SIMULATION_MIN_ALGORITHM_ITERATION_RUNS', 1)
    before = torch.rand(1)

    search.get_action(Model(), config)
    search.get_action(Model(), config)

    assert len(draws) == 6
    assert len(set(draws[:3])) == 1 and len(set(draws[3:])) == 1
    assert draws[0] != draws[3]  # a new search draws a new seed
    assert torch.rand(1).item() != before.item()  # caller's RNG keeps advancing normally


class MarkingScore:
    """Fake scorer whose 'training' appends 'trained' to the model, so tests can see what was trained."""

    def __init__(self):
        self.scored = []

    def score_and_train(self, gm, cfg):
        self.scored.append(list(gm))
        return 0.5, gm + ['trained']


def test_children_continue_from_the_parents_trained_model(monkeypatch):
    scorer = MarkingScore()
    config = SimpleNamespace(experiment_board=None, simulation_score=scorer)
    monkeypatch.setattr(search, 'generate_all_actions',
                        lambda model, cfg: [Action('x')] if model.gm == [] else [Action('y')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 0)
    root = search.TreeNode(None, None, Model(), config)
    root.expand()

    search._simulate(root)  # scores x
    search._simulate(root)  # expands x and scores x -> y

    assert scorer.scored == [['x'], ['x', 'trained', 'y']]
    assert root.traced.gm == []


def test_rollout_follow_ups_continue_from_the_trained_model(monkeypatch):
    scorer = MarkingScore()
    config = SimpleNamespace(experiment_board=None, simulation_score=scorer)
    monkeypatch.setattr(search, 'generate_all_actions', lambda model, cfg: [Action('next')])
    monkeypatch.setattr(search.project_config, 'MCTS_ROLLOUT_DEPTH', 2)
    node = search.TreeNode(None, None, Model(), config)

    node.rollout()

    assert scorer.scored == [[], ['trained', 'next'], ['trained', 'next', 'trained', 'next']]
    assert node.traced.gm == []
