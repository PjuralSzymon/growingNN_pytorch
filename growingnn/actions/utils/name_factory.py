"""Unique submodule names for FX ``call_module`` targets."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn


def unique_call_module_name(
    base: str,
    model: nn.Module | fx.GraphModule
) -> str:
    """Pick a non-colliding name for ``base``.

    If no existing child / graph target matches ``base`` or ``base_<int>``, returns
    ``base + "_0"``. Otherwise returns ``base + "_" + str(max_suffix + 1)``, where
    ``max_suffix`` is the largest integer among ``base`` (treated as 0) and
    ``base_1``, ``base_2``, … . If ``base`` exists without a numeric suffix, the
    next name is ``base_1``.

    """
    names: set[str] = set(model._modules.keys())
    if isinstance(model, fx.GraphModule):
        names |= {
            str(n.target) for n in model.graph.nodes if n.op == "call_module"
        }

    suffixes: list[int] = []
    for n in names:
        if n == base:
            suffixes.append(0)
        elif n.startswith(base + "_"):
            rest = n[len(base) + 1 :]
            if rest.isdigit():
                suffixes.append(int(rest))

    if not suffixes:
        return base + "_0"
    return base + "_" + str(max(suffixes) + 1)
