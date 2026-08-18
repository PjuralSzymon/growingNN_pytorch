"""UGapE-style gap-based best-arm identification on root actions (depth-1)."""

from __future__ import annotations

import copy
import math
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

# Exploration width scale for simple confidence bounds around noisy scores.
UGAPE_C = 1.0


def _bound(mean: float, n: int, total_pulls: int) -> float:
    if n <= 0:
        return float("inf")
    return mean + UGAPE_C * math.sqrt(max(math.log(max(total_pulls, 1)), 1.0) / n)


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

    rollouts = 0

    def pull(arm: dict) -> None:
        nonlocal rollouts
        value = score_fn(arm["child"].gm, running_config)
        rollouts += 1
        arm["n"] += 1
        arm["mean"] += (value - arm["mean"]) / arm["n"]

    # Base case: one pull per root action when time allows.
    for arm in arms:
        if time.time() >= deadline:
            break
        pull(arm)

    while time.time() < deadline:
        scored = [arm for arm in arms if arm["n"] > 0]
        if len(scored) < 2:
            for arm in arms:
                if arm["n"] == 0 and time.time() < deadline:
                    pull(arm)
                    break
            else:
                break
            continue
        best = max(scored, key=lambda item: item["mean"])
        challenger = max(
            (arm for arm in scored if arm is not best),
            key=lambda item: _bound(item["mean"], item["n"], rollouts),
        )
        # Pull the contested arm with fewer samples (gap uncertainty).
        chosen = best if best["n"] <= challenger["n"] else challenger
        pull(chosen)

    scored = [arm for arm in arms if arm["n"] > 0]
    best = max(scored, key=lambda item: item["mean"]) if scored else None
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
