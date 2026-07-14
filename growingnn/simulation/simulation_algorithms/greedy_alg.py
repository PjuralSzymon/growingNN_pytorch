"""Greedy random rollout architecture search within a time budget."""

from __future__ import annotations

import copy
import random
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def get_action(
    traced: TracedModel,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    all_actions = generate_all_actions(traced, running_config)
    if not all_actions:
        return None, 0, 0

    board = running_config.experiment_board
    params_before = traced.param_count() if board else None
    deadline = time.time() + running_config.simulation_scheduler.simulation_time
    t0 = time.time()
    best_action = None
    best_score = float("-inf")
    rollouts = 0
    remaining = list(all_actions)
    candidates: list[dict] = []

    while time.time() < deadline and remaining:
        action = random.choice(remaining)
        remaining.remove(action)
        candidate = copy.deepcopy(traced)
        action.execute(candidate)
        score = running_config.simulation_score.score(candidate.gm, running_config)
        rollouts += 1
        if board:
            candidates.append(
                board.greedy_candidate_row(action, candidate.gm, score, running_config, len(candidates))
            )
        else:
            candidates.append({"action": str(action), "score": score, "visits": 1, "ucbScore": score})
        if score > best_score:
            best_score = score
            best_action = action

    clear_reshepers_cache()
    if board and params_before is not None:
        board.finish_greedy_simulation(
            action=best_action,
            rollouts=rollouts,
            duration_sec=time.time() - t0,
            param_count_before=params_before,
            candidates=candidates,
        )

    return best_action, 0, rollouts
