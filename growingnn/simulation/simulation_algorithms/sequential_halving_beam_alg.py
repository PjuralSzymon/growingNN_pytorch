"""Sequential Halving on root actions, then beam deepen on survivors."""

from __future__ import annotations

import copy
import math
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

BEAM_WIDTH = 3
MAX_DEPTH = 2
# Fraction of wall time reserved for root Sequential Halving before beam deepen.
ROOT_TIME_FRACTION = 0.5


def get_action(
    traced: TracedModel,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    root_actions = generate_all_actions(traced, running_config)
    if not root_actions:
        return None, 0, 0

    board = running_config.experiment_board
    params_before = traced.param_count() if board else None
    total_time = running_config.simulation_scheduler.simulation_time
    t0 = time.time()
    root_deadline = t0 + total_time * ROOT_TIME_FRACTION
    deadline = t0 + total_time
    score_fn = running_config.simulation_score.score
    rollouts = 0
    max_depth = 0

    arms: list[dict] = []
    for action in root_actions:
        child = copy.deepcopy(traced)
        action.execute(child)
        arms.append({"action": action, "child": child, "mean": 0.0, "n": 0})

    # First pass: grade every root arm once (may overrun simulation_time).
    for arm in arms:
        value = score_fn(arm["child"].gm, running_config)
        rollouts += 1
        arm["n"] = 1
        arm["mean"] = value

    living = list(arms)
    while len(living) > 1 and time.time() < root_deadline:
        for arm in living:
            if time.time() >= root_deadline:
                break
            value = score_fn(arm["child"].gm, running_config)
            rollouts += 1
            arm["n"] += 1
            arm["mean"] += (value - arm["mean"]) / arm["n"]
        living.sort(key=lambda item: item["mean"], reverse=True)
        living = living[: max(1, math.ceil(len(living) / 2))]

    survivors = [arm for arm in living if arm["n"] > 0] or arms[:1]
    beam = [
        {
            "traced": arm["child"],
            "root_action": arm["action"],
            "score": arm["mean"],
            "depth": 1,
        }
        for arm in survivors[:BEAM_WIDTH]
    ]
    best = max(beam, key=lambda item: item["score"])
    max_depth = 1

    while time.time() < deadline and beam and beam[0]["depth"] < MAX_DEPTH:
        nxt: list[dict] = []
        for node in beam:
            if time.time() >= deadline:
                break
            for action in generate_all_actions(node["traced"], running_config):
                if time.time() >= deadline:
                    break
                child = copy.deepcopy(node["traced"])
                action.execute(child)
                value = score_fn(child.gm, running_config)
                rollouts += 1
                nxt.append(
                    {
                        "traced": child,
                        "root_action": node["root_action"],
                        "score": value,
                        "depth": node["depth"] + 1,
                    }
                )
        if not nxt:
            break
        nxt.sort(key=lambda item: item["score"], reverse=True)
        beam = nxt[:BEAM_WIDTH]
        max_depth = max(max_depth, beam[0]["depth"])
        if beam[0]["score"] > best["score"]:
            best = beam[0]

    best_action = best["root_action"]
    clear_reshepers_cache()

    if board and params_before is not None:
        candidates = [
            {
                "action": str(arm["action"]),
                "score": arm["mean"] if arm["n"] else None,
                "visits": arm["n"],
                "ucbScore": arm["mean"] if arm["n"] else None,
                "chosen": arm["action"] is best_action,
            }
            for arm in arms
        ]
        board.on_simulation_finished(
            getattr(board, "_current_generation", 0),
            action=best_action,
            max_depth=max_depth,
            rollouts=rollouts,
            duration_sec=time.time() - t0,
            param_count_before=params_before,
            candidates=candidates,
            search_tree=None,
        )

    return best_action, max_depth, rollouts
