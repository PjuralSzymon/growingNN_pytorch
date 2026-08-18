"""Progressive widening: unlock root actions over time, then Sequential Halving / beam."""

from __future__ import annotations

import copy
import math
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

INITIAL_OPEN = 3
UNLOCK_EVERY_SEC = 5.0
BEAM_WIDTH = 3
MAX_DEPTH = 2


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
    next_unlock = t0 + UNLOCK_EVERY_SEC
    score_fn = running_config.simulation_score.score
    rollouts = 0
    max_depth = 0

    # Fixed order unlock; no random sampling of the closed set.
    closed = list(all_actions)
    open_arms: list[dict] = []

    def unlock_one() -> None:
        if not closed:
            return
        action = closed.pop(0)
        child = copy.deepcopy(traced)
        action.execute(child)
        open_arms.append({"action": action, "child": child, "mean": 0.0, "n": 0})

    for _ in range(min(INITIAL_OPEN, len(closed))):
        unlock_one()

    best_action = None
    best_score = float("-inf")

    while time.time() < deadline and open_arms:
        if time.time() >= next_unlock and closed:
            unlock_one()
            next_unlock = time.time() + UNLOCK_EVERY_SEC

        # One Sequential Halving-style pull pass on the open set.
        living = list(open_arms)
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
            if time.time() >= next_unlock:
                break

        for arm in living:
            if arm["n"] == 0 and time.time() < deadline:
                value = score_fn(arm["child"].gm, running_config)
                rollouts += 1
                arm["n"] = 1
                arm["mean"] = value

        survivors = [arm for arm in living if arm["n"] > 0]
        if not survivors:
            break
        survivors.sort(key=lambda item: item["mean"], reverse=True)
        if survivors[0]["mean"] > best_score:
            best_score = survivors[0]["mean"]
            best_action = survivors[0]["action"]

        # Limited beam deepen from current open survivors if time remains.
        beam = [
            {
                "traced": arm["child"],
                "root_action": arm["action"],
                "score": arm["mean"],
                "depth": 1,
            }
            for arm in survivors[:BEAM_WIDTH]
        ]
        max_depth = max(max_depth, 1)
        if time.time() >= deadline or MAX_DEPTH <= 1:
            if closed:
                continue
            break

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
        if nxt:
            nxt.sort(key=lambda item: item["score"], reverse=True)
            top = nxt[0]
            max_depth = max(max_depth, top["depth"])
            if top["score"] > best_score:
                best_score = top["score"]
                best_action = top["root_action"]

        if not closed:
            break

    if best_action is None and open_arms:
        scored = [arm for arm in open_arms if arm["n"] > 0]
        if scored:
            best_action = max(scored, key=lambda item: item["mean"])["action"]

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
            for arm in open_arms
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
