"""Sequential Halving over root architecture actions (depth-1 only)."""

from __future__ import annotations

import copy
import math
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
    score_fn = running_config.simulation_score.score

    # One expand per root action; repeated pulls only rescore the same child.
    arms: list[dict] = []
    for action in all_actions:
        child = copy.deepcopy(traced)
        action.execute(child)
        arms.append(
            {
                "action": action,
                "child": child,
                "mean": 0.0,
                "n": 0,
            }
        )

    rollouts = 0
    living = list(arms)
    rounds = max(1, math.ceil(math.log2(len(living))))

    while len(living) > 1 and time.time() < deadline:
        pulls_each = 1
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
        living.sort(key=lambda item: item["mean"], reverse=True)
        keep = max(1, math.ceil(len(living) / 2))
        living = living[:keep]
        rounds = max(1, rounds - 1)

    best = max(living, key=lambda item: item["mean"]) if living else None
    best_action = best["action"] if best is not None else None
    clear_reshepers_cache()

    if board and params_before is not None:
        candidates = []
        for index, arm in enumerate(arms):
            row = {
                "action": str(arm["action"]),
                "score": arm["mean"] if arm["n"] else None,
                "visits": arm["n"],
                "ucbScore": arm["mean"] if arm["n"] else None,
                "chosen": arm["action"] is best_action,
            }
            if hasattr(board, "greedy_candidate_row"):
                row = board.greedy_candidate_row(
                    arm["action"], arm["child"].gm, arm["mean"] if arm["n"] else 0.0, running_config, index
                )
                row["visits"] = arm["n"]
                row["chosen"] = arm["action"] is best_action
            candidates.append(row)
        if hasattr(board, "finish_greedy_simulation"):
            board.finish_greedy_simulation(
                action=best_action,
                rollouts=rollouts,
                duration_sec=time.time() - t0,
                param_count_before=params_before,
                candidates=candidates,
            )
        else:
            board.on_simulation_finished(
                getattr(board, "_current_generation", 0),
                action=best_action,
                max_depth=0,
                rollouts=rollouts,
                duration_sec=time.time() - t0,
                param_count_before=params_before,
                candidates=candidates,
                search_tree=None,
            )

    return best_action, 0, rollouts
