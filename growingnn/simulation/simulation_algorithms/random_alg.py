"""Random architecture search baseline."""

from __future__ import annotations

import random
import time

import torch.fx as fx
import torch.nn as nn

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.utils.fx import GraphStructureQuery
from growingnn.utils.quaziIdentity import clear_reshepers_cache


def get_action(
    model: nn.Module | fx.GraphModule,
    running_config: RunningConfig,
) -> tuple[object | None, int, int]:
    actions = generate_all_actions(model, running_config)
    if not actions:
        return None, 0, 0

    board = running_config.experiment_board
    params_before = GraphStructureQuery.get_amount_of_parameters(model) if board else None
    t0 = time.time()
    action = random.choice(actions)
    clear_reshepers_cache()

    if board is not None and params_before is not None:
        action_str = str(action)
        candidates = [
            {
                "action": action_str,
                "name": board.action_short_label(action_str),
                "visits": 1,
                "score": None,
                "ucbScore": None,
                "compositeScore": None,
                "chosen": True,
            }
        ]
        board.on_simulation_finished(
            getattr(board, "_current_generation", 0),
            action=action,
            max_depth=0,
            rollouts=1,
            duration_sec=time.time() - t0,
            param_count_before=params_before,
            candidates=candidates,
            search_tree=board.search_tree_from_candidates(candidates, rollouts=1),
        )

    return action, 0, 0
