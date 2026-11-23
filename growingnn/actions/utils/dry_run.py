"""Check an architecture edit by running it on a copy of the network."""

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.fx as fx


def output_shapes(gm: fx.GraphModule, probe: torch.Tensor):
    """Shapes of the network's output(s) for *probe*, computed in evaluation mode."""
    was_training = gm.training
    gm.eval()
    try:
        with torch.no_grad():
            out = gm(probe)
    finally:
        gm.train(was_training)
    if isinstance(out, (tuple, list)):
        return tuple(tuple(item.shape) for item in out)
    return tuple(out.shape)


def edit_keeps_network_working(gm: fx.GraphModule, probe: torch.Tensor,
                               edit: Callable[[fx.GraphModule], object]) -> bool:
    """True when *edit* applied to a copy of *gm* succeeds and the output shapes stay the same.

    Width propagation sweeps the whole graph but cannot see through every operation (view, reshape,
    reductions, concatenation). Running the edit on a copy catches every case it gets wrong.
    """
    expected = output_shapes(gm, probe)
    candidate = copy.deepcopy(gm)
    try:
        edit(candidate)
        return output_shapes(candidate, probe) == expected
    except Exception:  # noqa: BLE001 - any failure means the edit breaks this network
        return False
