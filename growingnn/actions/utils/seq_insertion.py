"""Shared helpers for sequential layer insertion when shapes match."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.fx as fx
import torch.nn as nn

from growingnn.utils.fx import GraphStructureQuery, LayerShapeAnalyser


@dataclass(frozen=True)
class SeqInsertCandidate:
    from_id: str
    to_id: str
    shape: tuple[int, ...]


def iter_seq_shape_matched_pairs(
    gm: nn.Module | fx.GraphModule,
) -> Iterable[SeqInsertCandidate]:
    """Yield sequential module pairs whose probed output/input activation shapes match."""
    out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)
    in_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm)
    for from_id, to_id in GraphStructureQuery.module_sequential_pairs(gm):
        shape = out_shapes.get(from_id)
        if shape is not None and shape == in_shapes.get(to_id):
            yield SeqInsertCandidate(from_id, to_id, shape)
