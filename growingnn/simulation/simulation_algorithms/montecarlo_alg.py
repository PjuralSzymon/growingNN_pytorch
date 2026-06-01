"""Monte Carlo tree search over architecture mutations."""

from __future__ import annotations

import copy
import math
import random
import time

import torch.fx as fx
import torch.nn as nn

from growingnn.actions.registry import generate_all_actions
from growingnn.simulation.context import SimulationContext
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.utils.quaziIdentity import clear_reshepers_cache

UCB1_C = 2
ROLLOUT_DEPTH = 2
QUICK_LR = LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.0001)


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
        ctx: SimulationContext,
        simulation_score: SimulationScore,
    ):
        self.parent = parent
        self.action = action
        self.model = model
        self.ctx = ctx
        self.simulation_score = simulation_score
        self.child_nodes: list[TreeNode] = []
        self.value = 0.0
        self.visit_counter = 0
        self._cleaned = False

    def expand(self) -> None:
        for action in generate_all_actions(self.model):
            model_copy = copy.deepcopy(self.model)
            action.execute(model_copy)
            quick_ctx = SimulationContext(
                train_loader=self.ctx.train_loader,
                val_loader=self.ctx.val_loader,
                criterion=self.ctx.criterion,
                lr_scheduler=QUICK_LR,
                epochs=1,
            )
            gradient_descent(
                model_copy,
                quick_ctx.epochs,
                quick_ctx.train_loader,
                quick_ctx.val_loader,
                quick_ctx.criterion,
                quick_ctx.lr_scheduler,
                quiet=True,
            )
            self.child_nodes.append(
                TreeNode(self, action, model_copy, self.ctx, self.simulation_score)
            )

    def rollout(self) -> float:
        model_copy = copy.deepcopy(self.model)
        depth = ROLLOUT_DEPTH
        while depth > 0:
            actions = generate_all_actions(model_copy)
            if not actions:
                break
            chosen = random.choice(actions)
            chosen.execute(model_copy)
            quick_ctx = SimulationContext(
                train_loader=self.ctx.train_loader,
                val_loader=self.ctx.val_loader,
                criterion=self.ctx.criterion,
                lr_scheduler=QUICK_LR,
                epochs=1,
            )
            gradient_descent(
                model_copy,
                quick_ctx.epochs,
                quick_ctx.train_loader,
                quick_ctx.val_loader,
                quick_ctx.criterion,
                quick_ctx.lr_scheduler,
                quiet=True,
            )
            depth -= 1
        return self.simulation_score.score(model_copy, self.ctx)

    def get_best_child(self) -> TreeNode | None:
        if not self.child_nodes:
            return None

        def ucb1(node: TreeNode) -> float:
            if node.visit_counter == 0:
                return float("inf")
            return node.value + UCB1_C * _protected_divide(
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
        return node.value, depth, rollouts
    value, depth, rollouts = _simulate(child, depth + 1, rollouts)
    node.value += value
    node.visit_counter += 1
    return node.value, depth, rollouts


async def get_action(
    model: nn.Module | fx.GraphModule,
    max_time_for_dec: float,
    ctx: SimulationContext,
    simulation_score: SimulationScore,
) -> tuple[object | None, int, int]:
    actions = generate_all_actions(model)
    if not actions:
        return None, 0, 0

    root = TreeNode(None, None, model, ctx, simulation_score)
    deadline = time.time() + max_time_for_dec
    max_depth = 0
    rollouts = 0
    while time.time() < deadline or rollouts <= len(actions):
        _, max_depth, rollouts = _simulate(root, 0, rollouts)
        if time.time() >= deadline and rollouts > len(actions):
            break

    best_child = root.get_best_child()
    best_action = best_child.action if best_child is not None else None
    root.kill()
    clear_reshepers_cache()
    return best_action, max_depth, rollouts
