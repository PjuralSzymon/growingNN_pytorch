"""Beam search over architecture actions with selective deepen."""

from __future__ import annotations

import copy
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

BEAM_WIDTH = 3
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
    max_depth = 0

    # First pass: grade every root action once (may overrun simulation_time).
    frontier: list[dict] = []
    for action in root_actions:
        child = copy.deepcopy(traced)
        action.execute(child)
        value = score_fn(child.gm, running_config)
        rollouts += 1
        frontier.append(
            {
                "traced": child,
                "root_action": action,
                "score": value,
                "depth": 1,
            }
        )
    if not frontier:
        clear_reshepers_cache()
        return None, 0, 0

    frontier.sort(key=lambda item: item["score"], reverse=True)
    best = frontier[0]
    beam = frontier[:BEAM_WIDTH]
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
                "action": str(item["root_action"]),
                "score": item["score"],
                "visits": 1,
                "ucbScore": item["score"],
                "chosen": item["root_action"] is best_action,
            }
            for item in frontier[: max(BEAM_WIDTH, 5)]
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
