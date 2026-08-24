"""UGapE on root actions, then limited beam deepen on the top contested arms."""

from __future__ import annotations

import copy
import math
import time

from growingnn.actions.registry import generate_all_actions
import growingnn.core.config as project_config
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

UGAPE_C = 1.0
ROOT_TIME_FRACTION = 0.5
BEAM_WIDTH = 3
MAX_DEPTH = 2
RIVAL_COUNT = 2


def _bound(mean: float, n: int, total_pulls: int) -> float:
    if n <= 0:
        return float("inf")
    return mean + UGAPE_C * math.sqrt(max(math.log(max(total_pulls, 1)), 1.0) / n)


def get_action(
    traced: TracedModel,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    root_actions = generate_all_actions(traced, running_config)
    if not root_actions:
        return None, 0, 0

    board = running_config.experiment_board
    params_before = traced.param_count() if board else None
    min_runs = project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS
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

    def pull(arm: dict) -> None:
        nonlocal rollouts
        value = score_fn(arm["child"].gm, running_config)
        rollouts += 1
        arm["n"] += 1
        arm["mean"] += (value - arm["mean"]) / arm["n"]

    # First pass: grade every root arm once (may overrun simulation_time).
    for arm in arms:
        pull(arm)

    extra_pulls = 0
    while extra_pulls < min_runs or time.time() < root_deadline:
        scored = [arm for arm in arms if arm["n"] > 0]
        if len(scored) < 2:
            break
        best = max(scored, key=lambda item: item["mean"])
        challenger = max(
            (arm for arm in scored if arm is not best),
            key=lambda item: _bound(item["mean"], item["n"], rollouts),
        )
        pull(best if best["n"] <= challenger["n"] else challenger)
        extra_pulls += 1

    scored = [arm for arm in arms if arm["n"] > 0]
    if not scored:
        clear_reshepers_cache()
        return None, 0, 0
    scored.sort(key=lambda item: item["mean"], reverse=True)
    focus = scored[: max(1, RIVAL_COUNT)]
    beam = [
        {
            "traced": arm["child"],
            "root_action": arm["action"],
            "score": arm["mean"],
            "depth": 1,
        }
        for arm in focus
    ]
    best_node = beam[0]
    max_depth = 1

    deepen_rounds = 0
    while beam and beam[0]["depth"] < MAX_DEPTH and (
        deepen_rounds < min_runs or time.time() < deadline
    ):
        required = deepen_rounds < min_runs
        nxt: list[dict] = []
        for node in beam:
            if not required and time.time() >= deadline:
                break
            for action in generate_all_actions(node["traced"], running_config):
                if not required and time.time() >= deadline:
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
        if beam[0]["score"] > best_node["score"]:
            best_node = beam[0]
        deepen_rounds += 1

    best_action = best_node["root_action"]
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
