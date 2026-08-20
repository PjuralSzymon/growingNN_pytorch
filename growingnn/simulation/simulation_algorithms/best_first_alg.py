"""Best-first search over architecture actions with a depth cap."""

from __future__ import annotations

import copy
import heapq
import time

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.utils.quaziIdentity import clear_reshepers_cache

MAX_DEPTH = 2
MAX_EXPANSIONS = 32
# Prefer slightly shallower nodes when scores are close.
DEPTH_PENALTY = 1e-3


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
    expansions = 0
    counter = 0
    open_heap: list[tuple[float, int, dict]] = []
    best: dict | None = None
    root_rows: list[dict] = []

    for action in root_actions:
        if time.time() >= deadline:
            break
        child = copy.deepcopy(traced)
        action.execute(child)
        value = score_fn(child.gm, running_config)
        rollouts += 1
        node = {
            "traced": child,
            "root_action": action,
            "score": value,
            "depth": 1,
        }
        root_rows.append(node)
        priority = -(value - DEPTH_PENALTY * node["depth"])
        heapq.heappush(open_heap, (priority, counter, node))
        counter += 1
        if best is None or value > best["score"]:
            best = node
        max_depth = 1

    while open_heap and time.time() < deadline and expansions < MAX_EXPANSIONS:
        _, _, node = heapq.heappop(open_heap)
        if node["depth"] >= MAX_DEPTH:
            continue
        expansions += 1
        for action in generate_all_actions(node["traced"], running_config):
            if time.time() >= deadline:
                break
            child = copy.deepcopy(node["traced"])
            action.execute(child)
            value = score_fn(child.gm, running_config)
            rollouts += 1
            child_node = {
                "traced": child,
                "root_action": node["root_action"],
                "score": value,
                "depth": node["depth"] + 1,
            }
            max_depth = max(max_depth, child_node["depth"])
            priority = -(value - DEPTH_PENALTY * child_node["depth"])
            heapq.heappush(open_heap, (priority, counter, child_node))
            counter += 1
            if best is None or value > best["score"]:
                best = child_node

    best_action = best["root_action"] if best is not None else None
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
            for item in root_rows[:8]
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
