from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from growingnn.simulation.simulation_algorithms import budgeted_mcts_alg as search


def test_budgeted_search_returns_only_evaluated_action_at_zero_budget():
    class Action:
        def execute(self, model):
            model.choice = self

    actions = [Action(), Action()]
    model = SimpleNamespace()
    config = SimpleNamespace(experiment_board=None,
                             simulation_scheduler=SimpleNamespace(simulation_time=0))
    with (patch.object(search, 'generate_all_actions', return_value=actions),
          patch.object(search.TreeNode, 'rollout', return_value=0.75),
          patch.object(search, 'clear_reshepers_cache')):
        action, depth, rollouts = search.get_action(model, config)
    assert action in actions
    assert depth == 1
    assert rollouts == 1
    assert not hasattr(model, 'choice')


def test_no_legal_action_returns_without_rollout():
    config = SimpleNamespace(experiment_board=None,
                             simulation_scheduler=SimpleNamespace(simulation_time=1))
    with (patch.object(search, 'generate_all_actions', return_value=[]),
          patch.object(search.TreeNode, 'rollout') as rollout,
          patch.object(search, 'clear_reshepers_cache')):
        assert search.get_action(SimpleNamespace(), config) == (None, 0, 0)
    rollout.assert_not_called()


def test_final_choice_uses_observed_mean():
    class Action:
        def __init__(self, score):
            self.score = score

        def execute(self, model):
            model.score = self.score

    actions = [Action(0.2), Action(0.8)]
    config = SimpleNamespace(experiment_board=None,
                             simulation_scheduler=SimpleNamespace(simulation_time=1))
    with (patch.object(search, 'generate_all_actions', return_value=actions),
          patch.object(search.random, 'shuffle'),
          patch.object(search.TreeNode, 'rollout', lambda node: node.traced.score),
          patch.object(search.time, 'monotonic', side_effect=[0, 0, 2]),
          patch.object(search, 'clear_reshepers_cache')):
        action, depth, rollouts = search.get_action(SimpleNamespace(), config)
    assert action is actions[1]
    assert depth == 1
    assert rollouts == 2


def test_board_marks_recommended_action_instead_of_exploration_winner():
    class Action:
        def __init__(self, score):
            self.score = score

        def execute(self, model):
            model.score = self.score

        def __str__(self):
            return str(self.score)

    actions = [Action(0.2), Action(0.8)]
    board = MagicMock()
    board._current_generation = 0
    board.mcts_candidates_from_root.return_value = [
        {'action': '0.2', 'score': 0.2, 'chosen': True},
        {'action': '0.8', 'score': 0.8, 'chosen': False},
    ]
    model = SimpleNamespace(param_count=lambda: 10)
    config = SimpleNamespace(experiment_board=board,
                             simulation_scheduler=SimpleNamespace(simulation_time=1))
    with (patch.object(search, 'generate_all_actions', return_value=actions),
          patch.object(search.random, 'shuffle'),
          patch.object(search.TreeNode, 'rollout', lambda node: node.traced.score),
          patch.object(search.time, 'monotonic', side_effect=[0, 0, 2, 2]),
          patch.object(search, 'clear_reshepers_cache')):
        search.get_action(model, config)
    rows = board.on_simulation_finished.call_args.kwargs['candidates']
    assert [row['chosen'] for row in rows] == [False, True]
    assert rows[1]['selectionScore'] == 0.8


def test_tree_depth_limit_rescores_root_action_without_deepening():
    action = MagicMock()
    config = SimpleNamespace(experiment_board=None, MCTS_MAX_TREE_DEPTH=1,
                             simulation_scheduler=SimpleNamespace(simulation_time=1))
    with (patch.object(search, 'generate_all_actions', return_value=[action]) as generate,
          patch.object(search.TreeNode, 'rollout', return_value=0.5),
          patch.object(search.time, 'monotonic', side_effect=[0, 0, 2]),
          patch.object(search, 'clear_reshepers_cache')):
        chosen, depth, rollouts = search.get_action(SimpleNamespace(), config)
    assert chosen is action
    assert (depth, rollouts) == (1, 2)
    assert generate.call_count == 1
