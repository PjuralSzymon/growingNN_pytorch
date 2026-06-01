"""Greedy random rollout architecture search within a time budget."""

from __future__ import annotations

import copy
import random
import time

import torch.fx as fx
import torch.nn as nn

from growingnn.actions.registry import generate_all_actions
from growingnn.simulation.context import SimulationContext
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.utils.quaziIdentity import clear_reshepers_cache


async def get_action(
    model: nn.Module | fx.GraphModule,
    max_time_for_dec: float,
    ctx: SimulationContext,
    simulation_score: SimulationScore,
) -> tuple[object | None, int, int]:
    all_actions = generate_all_actions(model)
    if not all_actions:
        return None, 0, 0

    deadline = time.time() + max_time_for_dec
    best_action = None
    best_score = float("-inf")
    rollouts = 0
    remaining = list(all_actions)

    while time.time() < deadline and remaining:
        action = random.choice(remaining)
        remaining.remove(action)
        candidate = copy.deepcopy(model)
        action.execute(candidate)
        score = simulation_score.score(candidate, ctx)
        rollouts += 1
        if score > best_score:
            best_score = score
            best_action = action

    clear_reshepers_cache()
    return best_action, 0, rollouts
