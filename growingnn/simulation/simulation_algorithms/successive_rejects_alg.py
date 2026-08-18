"""Successive Rejects over root architecture actions (depth-1 only)."""

from __future__ import annotations

import copy
import math
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def _log_bar(n: int) -> float:
    # Audibert et al.: 1/2 + sum_{i=2..n} 1/i
    if n <= 1:
        return 0.5
    return 0.5 + sum(1.0 / i for i in range(2, n + 1))


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
    score_fn = running_config.simulation_score.score

    arms: list[dict] = []
    for action in all_actions:
        child = copy.deepcopy(traced)
        action.execute(child)
        arms.append({"action": action, "child": child, "mean": 0.0, "n": 0})

    living = list(arms)
    n = len(living)
    log_bar = _log_bar(n)
    # Approximate pull budget from remaining wall time with a soft cap.
    estimated_pull_budget = max(n * 2, n * max(1, int(running_config.simulation_scheduler.simulation_time // 5)))
    rollouts = 0
    prev_nk = 0

    for round_idx in range(1, n):
        if time.time() >= deadline or len(living) <= 1:
            break
        remaining_arms = n + 1 - round_idx
        nk = int(math.ceil((1.0 / log_bar) * ((n - round_idx) / remaining_arms) * estimated_pull_budget))
        pulls_each = max(1, nk - prev_nk)
        prev_nk = nk
        for arm in living:
            if time.time() >= deadline:
                break
            for _ in range(pulls_each):
                if time.time() >= deadline:
                    break
                value = score_fn(arm["child"].gm, running_config)
                rollouts += 1
                arm["n"] += 1
                arm["mean"] += (value - arm["mean"]) / arm["n"]
        if len(living) <= 1:
            break
        living.sort(key=lambda item: item["mean"], reverse=True)
        living = living[:-1]  # reject current worst

    best = max(living, key=lambda item: item["mean"]) if living else None
    best_action = best["action"] if best is not None else None
    clear_reshepers_cache()

    if board and params_before is not None and hasattr(board, "finish_greedy_simulation"):
        candidates = [
            board.greedy_candidate_row(
                arm["action"],
                arm["child"].gm,
                arm["mean"] if arm["n"] else 0.0,
                running_config,
                index,
            )
            for index, arm in enumerate(arms)
        ]
        for arm, row in zip(arms, candidates):
            row["visits"] = arm["n"]
            row["chosen"] = arm["action"] is best_action
        board.finish_greedy_simulation(
            action=best_action,
            rollouts=rollouts,
            duration_sec=time.time() - t0,
            param_count_before=params_before,
            candidates=candidates,
        )

    return best_action, 0, rollouts
