"""Random architecture search baseline."""

from __future__ import annotations

import random

import torch.fx as fx
import torch.nn as nn

from growingnn.actions.registry import generate_all_actions
from growingnn.simulation.context import SimulationContext
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.utils.quaziIdentity import clear_reshepers_cache


async def get_action(
    model: nn.Module | fx.GraphModule,
    max_time_for_dec: float,
    ctx: SimulationContext,
    simulation_score: SimulationScore | None = None,
) -> tuple[object | None, int, int]:
    del max_time_for_dec, ctx, simulation_score
    actions = generate_all_actions(model)
    if not actions:
        return None, 0, 0
    clear_reshepers_cache()
    return random.choice(actions), 0, 0
