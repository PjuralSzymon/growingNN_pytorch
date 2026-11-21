"""Monte Carlo tree search over architecture mutations."""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Any

import torch

from growingnn.actions.registry import generate_all_actions
import growingnn.core.config as project_config
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.core.logger import logger
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def _protected_divide(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return a / b


def _group_by_type(actions: list) -> list[list]:
    groups: dict[type, list] = {}
    for action in actions:
        groups.setdefault(type(action), []).append(action)
    return list(groups.values())


def _type_interleaved(actions: list) -> list:
    """Random order that cycles through action types, so no single type fills the front."""
    groups = _group_by_type(actions)
    for group in groups:
        random.shuffle(group)
    random.shuffle(groups)
    longest = max((len(group) for group in groups), default=0)
    return [group[i] for i in range(longest) for group in groups if i < len(group)]


def _random_follow_up(actions: list):
    """Pick an action type uniformly, then an action of that type."""
    return random.choice(random.choice(_group_by_type(actions)))


class TreeNode:
    def __init__(
        self,
        parent: TreeNode | None,
        action,
        traced: TracedModel,
        running_config: RunningConfig,
    ):
        self.parent = parent
        self.action = action
        self.traced = traced
        self.running_config = running_config
        self.child_nodes: list[TreeNode] = []
        self.value = 0.0
        self.visit_counter = 0
        self._cleaned = False
        self.rollout_metrics: dict[str, Any] | None = None
        self._own_score: float | None = None  # this node's model never changes, so score it once
        # Every scoring run in one search uses the same training seed, so score differences
        # come from the architectures rather than from the random training draw.
        self.search_seed: int | None = parent.search_seed if parent is not None else None

    def _score(self, traced: TracedModel) -> float:
        cfg = self.running_config
        if self.search_seed is None:
            return cfg.simulation_score.score(copy.deepcopy(traced.gm), cfg)
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(self.search_seed)
            return cfg.simulation_score.score(copy.deepcopy(traced.gm), cfg)

    def expand(self) -> None:
        cfg = self.running_config
        for action in _type_interleaved(generate_all_actions(self.traced, cfg)):
            traced_copy = copy.deepcopy(self.traced)
            action.execute(traced_copy)
            self.child_nodes.append(
                TreeNode(self, action, traced_copy, cfg)
            )

    def rollout(self) -> float:
        """Score this node's model, then up to MCTS_ROLLOUT_DEPTH random follow-up actions.

        Returns the average of the scores weighted 1, decay, decay**2, ...
        """
        cfg = self.running_config
        traced_copy = copy.deepcopy(self.traced)
        depth = project_config.MCTS_ROLLOUT_DEPTH
        if self._own_score is None:
            self._own_score = self._score(traced_copy)
        weight = 1.0
        weighted_score = self._own_score
        total_weight = weight
        while depth > 0:
            actions = generate_all_actions(traced_copy, cfg)
            if not actions:
                break
            chosen = _random_follow_up(actions)
            chosen.execute(traced_copy)
            weight *= project_config.MCTS_ROLLOUT_WEIGHT_DECAY
            weighted_score += weight * self._score(traced_copy)
            total_weight += weight
            depth -= 1
        composite = weighted_score / total_weight
        if cfg.experiment_board is not None:
            self.rollout_metrics = dict(cfg.experiment_board.simulation_metrics)
            self.rollout_metrics["composite_score"] = composite
        return composite

    def get_best_child(self) -> TreeNode | None:
        if not self.child_nodes:
            return None

        def ucb1(node: TreeNode) -> float:
            if node.visit_counter == 0:
                return float("inf")
            if project_config.MCTS_UCB1_USE_SQRT:
                n = node.visit_counter
                explore = math.sqrt(math.log(max(self.visit_counter, 1)) / n)
                return node.value / n + project_config.MCTS_UCB1_C * explore
            return node.value / node.visit_counter + project_config.MCTS_UCB1_C * _protected_divide(
                math.log(max(self.visit_counter, 1)),
                node.visit_counter,
            )

        return max(self.child_nodes, key=ucb1)

    def get_final_child(self) -> TreeNode | None:
        """Choose the evaluated child with the highest mean search value."""
        evaluated = (child for child in self.child_nodes if child.visit_counter > 0)
        return max(evaluated, key=lambda child: child.value / child.visit_counter, default=None)

    def is_leaf(self) -> bool:
        return len(self.child_nodes) == 0

    def kill(self) -> None:
        if self._cleaned:
            return
        for child in self.child_nodes:
            child.kill()
        self.child_nodes.clear()
        self.traced = None
        self.parent = None
        self._cleaned = True


def _simulate(node: TreeNode, depth: int = 0, rollouts: int = 0) -> tuple[float, int, int]:
    if node.is_leaf():
        if node.visit_counter == 0:
            value = node.rollout()
            node.value = value
            node.visit_counter += 1
            return value, depth, rollouts + 1
        node.expand()
    child = node.get_best_child()
    if child is None:
        return node.value / node.visit_counter, depth, rollouts
    value, depth, rollouts = _simulate(child, depth + 1, rollouts)
    node.value += value
    node.visit_counter += 1
    return value, depth, rollouts


def get_action(
    traced: TracedModel,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    actions = generate_all_actions(traced, running_config)
    if not actions:
        logger.warning("No actions generated for model")
        return None, 0, 0

    board = running_config.experiment_board
    params_before = traced.param_count() if board is not None else None

    root = TreeNode(None, None, traced, running_config)
    root.search_seed = random.randrange(2**32)
    min_runs = project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS
    deadline = time.time() + running_config.simulation_scheduler.simulation_time
    t0 = time.time()
    max_depth = 0
    rollouts = 0
    simulate_calls = 0
    root.expand()
    while (
        time.time() < deadline
        or rollouts < len(actions)
        or simulate_calls < min_runs
    ):
        prev_rollouts = rollouts
        _, max_depth, rollouts = _simulate(root, 0, rollouts)
        simulate_calls += 1
        if time.time() >= deadline:
            if rollouts >= len(actions) and simulate_calls >= min_runs:
                break
            elif rollouts <= prev_rollouts:
                logger.error("MCTS no new rollouts after deadline (rollouts=%s, actions=%s)",rollouts,len(actions),)

    best_child = root.get_final_child()
    best_action = best_child.action if best_child is not None else None
    if best_action is None:
        logger.warning("No best action found for MCTS simulation")
    candidates = None
    search_tree = None
    generation = getattr(board, "_current_generation", 0) if board is not None else 0
    if board is not None:
        candidates = board.mcts_candidates_from_root(root, running_config)
        for row in candidates:
            row["chosen"] = best_action is not None and row.get("action") == str(best_action)
            row["selectionScore"] = row.get("score")
        search_tree = board.mcts_search_tree_from_root(
            root, running_config, chosen_node=best_child, max_depth=max_depth
        )
    root.kill()
    clear_reshepers_cache()

    if board is not None and params_before is not None:
        board.on_simulation_finished(
            generation,
            action=best_action,
            max_depth=max_depth,
            rollouts=rollouts,
            duration_sec=time.time() - t0,
            param_count_before=params_before,
            candidates=candidates,
            search_tree=search_tree,
        )

    return best_action, max_depth, rollouts
