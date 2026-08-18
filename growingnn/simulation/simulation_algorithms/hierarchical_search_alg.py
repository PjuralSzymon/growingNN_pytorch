"""Hierarchical search: rank action families, then Sequential Halving + beam inside winners."""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

TOP_FAMILIES = 2
BEAM_WIDTH = 3
MAX_DEPTH = 2
FAMILY_TIME_FRACTION = 0.25


def _family_key(action: object) -> str:
    return type(action).__name__


def get_action(
    traced: TracedModel,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    all_actions = generate_all_actions(traced, running_config)
    if not all_actions:
        return None, 0, 0

    board = running_config.experiment_board
    params_before = traced.param_count() if board else None
    total_time = running_config.simulation_scheduler.simulation_time
    t0 = time.time()
    family_deadline = t0 + total_time * FAMILY_TIME_FRACTION
    deadline = t0 + total_time
    score_fn = running_config.simulation_score.score
    rollouts = 0
    max_depth = 0

    by_family: dict[str, list] = defaultdict(list)
    for action in all_actions:
        by_family[_family_key(action)].append(action)

    family_scores: list[tuple[str, float]] = []
    for family, actions in by_family.items():
        if time.time() >= family_deadline:
            break
        # One sample action grades the family.
        sample = actions[0]
        child = copy.deepcopy(traced)
        sample.execute(child)
        value = score_fn(child.gm, running_config)
        rollouts += 1
        family_scores.append((family, value))

    if not family_scores:
        # Time ran out before any family sample; fall back to first families in map order.
        chosen_families = list(by_family.keys())[:TOP_FAMILIES]
    else:
        family_scores.sort(key=lambda item: item[1], reverse=True)
        chosen_families = [name for name, _ in family_scores[:TOP_FAMILIES]]

    pool_actions = []
    for family in chosen_families:
        pool_actions.extend(by_family[family])
    if not pool_actions:
        pool_actions = list(all_actions)

    arms: list[dict] = []
    for action in pool_actions:
        if time.time() >= deadline and arms:
            break
        child = copy.deepcopy(traced)
        action.execute(child)
        arms.append({"action": action, "child": child, "mean": 0.0, "n": 0})

    living = list(arms)
    while len(living) > 1 and time.time() < deadline:
        for arm in living:
            if time.time() >= deadline:
                break
            value = score_fn(arm["child"].gm, running_config)
            rollouts += 1
            arm["n"] += 1
            arm["mean"] += (value - arm["mean"]) / arm["n"]
        living.sort(key=lambda item: item["mean"], reverse=True)
        living = living[: max(1, math.ceil(len(living) / 2))]

    for arm in living:
        if arm["n"] == 0 and time.time() < deadline:
            value = score_fn(arm["child"].gm, running_config)
            rollouts += 1
            arm["n"] = 1
            arm["mean"] = value

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
