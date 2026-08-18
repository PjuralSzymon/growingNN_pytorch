"""SHOT: Sequential Halving applied recursively on a depth-limited action tree."""

from __future__ import annotations

import copy
import math
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

MAX_DEPTH = 2


def get_action(
    traced: TracedModel,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    root_actions = generate_all_actions(traced, running_config)
    if not root_actions:
        return None, 0, 0

    board = running_config.experiment_board
    params_before = traced.param_count() if board else None
    deadline = time.time() + running_config.simulation_scheduler.simulation_time
    t0 = time.time()
    score_fn = running_config.simulation_score.score
    rollouts = 0
    max_depth_seen = 0

    def score_node(node_traced: TracedModel) -> float:
        nonlocal rollouts
        value = score_fn(node_traced.gm, running_config)
        rollouts += 1
        return value

    def shot(node_traced: TracedModel, depth_left: int) -> tuple[float, object | None]:
        nonlocal max_depth_seen
        if depth_left <= 0 or time.time() >= deadline:
            return score_node(node_traced), None

        actions = generate_all_actions(node_traced, running_config)
        if not actions:
            return score_node(node_traced), None

        arms: list[dict] = []
        for action in actions:
            if time.time() >= deadline:
                break
            child = copy.deepcopy(node_traced)
            action.execute(child)
            arms.append({"action": action, "child": child, "value": float("-inf"), "n": 0})

        if not arms:
            return score_node(node_traced), None

        living = list(arms)
        while len(living) > 1 and time.time() < deadline:
            for arm in living:
                if time.time() >= deadline:
                    break
                value, _ = shot(arm["child"], depth_left - 1)
                max_depth_seen = max(max_depth_seen, MAX_DEPTH - depth_left + 1)
                arm["n"] += 1
                if arm["value"] == float("-inf"):
                    arm["value"] = value
                else:
                    arm["value"] += (value - arm["value"]) / arm["n"]
            living.sort(key=lambda item: item["value"], reverse=True)
            living = living[: max(1, math.ceil(len(living) / 2))]

        best_arm = max(living, key=lambda item: item["value"])
        return best_arm["value"], best_arm["action"]

    # Root SHOT: recommend the best root child after recursive halving.
    root_arms: list[dict] = []
    for action in root_actions:
        if time.time() >= deadline and root_arms:
            break
        child = copy.deepcopy(traced)
        action.execute(child)
        root_arms.append({"action": action, "child": child, "value": float("-inf"), "n": 0})

    living = list(root_arms)
    while len(living) > 1 and time.time() < deadline:
        for arm in living:
            if time.time() >= deadline:
                break
            value, _ = shot(arm["child"], MAX_DEPTH - 1)
            max_depth_seen = max(max_depth_seen, 1)
            arm["n"] += 1
            if arm["value"] == float("-inf"):
                arm["value"] = value
            else:
                arm["value"] += (value - arm["value"]) / arm["n"]
        living.sort(key=lambda item: item["value"], reverse=True)
        living = living[: max(1, math.ceil(len(living) / 2))]

    # Ensure every remaining root arm has at least one value if time remains.
    for arm in living:
        if arm["n"] == 0 and time.time() < deadline:
            value, _ = shot(arm["child"], MAX_DEPTH - 1)
            arm["value"] = value
            arm["n"] = 1

    scored = [arm for arm in living if arm["n"] > 0] or [arm for arm in root_arms if arm["n"] > 0]
    best = max(scored, key=lambda item: item["value"]) if scored else None
    best_action = best["action"] if best is not None else None
    clear_reshepers_cache()

    if board and params_before is not None:
        candidates = [
            {
                "action": str(arm["action"]),
                "score": arm["value"] if arm["n"] else None,
                "visits": arm["n"],
                "ucbScore": arm["value"] if arm["n"] else None,
                "chosen": arm["action"] is best_action,
            }
            for arm in root_arms
        ]
        board.on_simulation_finished(
            getattr(board, "_current_generation", 0),
            action=best_action,
            max_depth=max_depth_seen,
            rollouts=rollouts,
            duration_sec=time.time() - t0,
            param_count_before=params_before,
            candidates=candidates,
            search_tree=None,
        )

    return best_action, max_depth_seen, rollouts
