"""Shared helpers for sequential layer insertion when shapes match."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from growingnn.core.traced_model import TracedModel


@dataclass(frozen=True)
class SeqInsertCandidate:
    from_id: str
    to_id: str
    shape: tuple[int, ...]


def iter_seq_shape_matched_pairs(traced: TracedModel) -> Iterable[SeqInsertCandidate]:
    """Yield sequential module pairs whose probed output/input activation shapes match."""
    out_shapes, in_shapes = traced.shapes()
    for from_id, to_id in traced.sequential_pairs():
        shape = out_shapes.get(from_id)
        if shape is not None and shape == in_shapes.get(to_id):
            yield SeqInsertCandidate(from_id, to_id, shape)
