"""Random architecture search baseline."""

from __future__ import annotations

import random

import torch.fx as fx
import torch.nn as nn

from growingnn.actions.registry import generate_all_actions
from growingnn.simulation.context import SimulationContext
from growingnn.utils.quaziIdentity import clear_reshepers_cache


async def get_action(
    model: nn.Module | fx.GraphModule,
    ctx: SimulationContext,
) -> tuple[object | None, int, int]:
    actions = generate_all_actions(model, ctx.running_config)
    if not actions:
        return None, 0, 0
    clear_reshepers_cache()
    return random.choice(actions), 0, 0
