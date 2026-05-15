"""FX helpers for tensor shapes on ``call_module`` nodes (``ShapeProp``)."""

from __future__ import annotations

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


def _node_output_shape_tuple(node: fx.Node) -> tuple[int, ...] | None:
    meta = node.meta
    val = meta.get("val")
    if val is not None and hasattr(val, "shape"):
        return tuple(int(x) for x in val.shape)
    tm = meta.get("tensor_meta")
    if tm is not None and hasattr(tm, "shape"):
        return tuple(int(x) for x in tm.shape)
    return None


def _default_probe_tensor(gm: fx.GraphModule) -> torch.Tensor | None:
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if not placeholders:
        return None
    try:
        p0 = next(gm.parameters())
        device, dtype = p0.device, p0.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    return torch.randn(1, 3, 224, 224, device=device, dtype=dtype)


def call_module_output_shapes(
    gm: fx.GraphModule,
    example: torch.Tensor | None = None,
) -> dict[str, tuple[int, ...]]:
    """Run ``ShapeProp`` and map each ``call_module`` target string to its output shape.

    ``add_new_residual_layer`` sums tensors at two module outputs; this map is used to
    reject pairs whose activations cannot broadcast (e.g. different spatial sizes
    across ResNet stages).

    Returns an empty dict if propagation fails or no probe tensor could be built.
    """
    probe = example if example is not None else _default_probe_tensor(gm)
    if probe is None:
        return {}
    try:
        ShapeProp(gm).propagate(probe)
    except Exception:
        return {}
    out: dict[str, tuple[int, ...]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(node.target, str):
            continue
        shape = _node_output_shape_tuple(node)
        if shape is not None:
            out[node.target] = shape
    return out
