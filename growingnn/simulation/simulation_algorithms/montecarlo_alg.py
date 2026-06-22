"""Monte Carlo tree search over architecture mutations."""

from __future__ import annotations

import copy
import math
import random
import time

import torch.fx as fx
import torch.nn as nn

import growingnn.core.config as project_config
from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.logger import logger
from growingnn.training.gradient_descent import gradient_descent
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def _protected_divide(a: float, b: float) -> float:
    if b == 0:
        return float("inf")
    return a / b

class TreeNode:
    def __init__(
        self,
        parent: TreeNode | None,
        action,
        model: nn.Module | fx.GraphModule,
        running_config: RunningConfig,
    ):
        self.parent = parent
        self.action = action
        self.model = model
        self.running_config = running_config
        self.child_nodes: list[TreeNode] = []
        self.value = 0.0
        self.visit_counter = 0
        self._cleaned = False

    def expand(self) -> None:
        cfg = self.running_config
        for action in generate_all_actions(self.model, cfg):
            model_copy = copy.deepcopy(self.model)
            action.execute(model_copy)
            try:
                gradient_descent(
                    model_copy,
                    project_config.MCTS_ROLLOUT_EPOCHS,
                    cfg.sim_train_loader,
                    cfg.sim_val_loader,
                    cfg.criterion,
                    project_config.MCTS_ROLLOUT_LR,
                    quiet=True,
                )
            except Exception as e:
                logger.error("Error in gradient_descent: %s", e)
                draw_filtered_fx_graph(model_copy, "fx_graph_error_simulation_simplified", fmt="pdf")
                exit(1)
            self.child_nodes.append(
                TreeNode(self, action, model_copy, cfg)
            )

    def rollout(self) -> float:
        cfg = self.running_config
        model_copy = copy.deepcopy(self.model)
        depth = project_config.MCTS_ROLLOUT_DEPTH
        while depth > 0:
            actions = generate_all_actions(model_copy, cfg)
            if not actions:
                break
            chosen = random.choice(actions)
            chosen.execute(model_copy)
            gradient_descent(
                model_copy,
                project_config.MCTS_ROLLOUT_EPOCHS,
                cfg.sim_train_loader,
                cfg.sim_val_loader,
                cfg.criterion,
                project_config.MCTS_ROLLOUT_LR,
                quiet=True,
            )
            depth -= 1
        return cfg.simulation_score.score(model_copy, cfg)

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
            return node.value + project_config.MCTS_UCB1_C * _protected_divide(
                math.log(max(self.visit_counter, 1)),
                node.visit_counter,
            )

        return max(self.child_nodes, key=ucb1)

    def is_leaf(self) -> bool:
        return len(self.child_nodes) == 0

    def kill(self) -> None:
        if self._cleaned:
            return
        for child in self.child_nodes:
            child.kill()
        self.child_nodes.clear()
        self.model = None  # type: ignore[assignment]
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
        backprop = 0.0 if project_config.MCTS_PROPAGATE_ROLLOUT_VALUE else node.value
        return backprop, depth, rollouts
    value, depth, rollouts = _simulate(child, depth + 1, rollouts)
    node.value += value
    node.visit_counter += 1
    if project_config.MCTS_PROPAGATE_ROLLOUT_VALUE:
        return value, depth, rollouts
    return node.value, depth, rollouts


async def get_action(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    actions = generate_all_actions(model, running_config)
    if not actions:
        return None, 0, 0

    root = TreeNode(None, None, model, running_config)
    deadline = time.time() + running_config.simulation_scheduler.simulation_time
    max_depth = 0
    rollouts = 0
    while time.time() < deadline or rollouts <= len(actions):
        prev_rollouts = rollouts
        _, max_depth, rollouts = _simulate(root, 0, rollouts)
        if time.time() >= deadline:
            if rollouts > len(actions):
                break
            elif rollouts <= prev_rollouts:
                logger.error("MCTS no new rollouts after deadline (rollouts=%s, actions=%s)",rollouts,len(actions),)

    best_child = root.get_best_child()
    best_action = best_child.action if best_child is not None else None
    root.kill()
    clear_reshepers_cache()
    return best_action, max_depth, rollouts
