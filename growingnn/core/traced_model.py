"""Traced FX model plus cached analysis for training and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.fx as fx
import torch.nn as nn

from growingnn.utils.fx.graph_analysis import GraphStructureQuery, LayerShapeAnalyser
from growingnn.utils.fx.graph_extraction import extract_graph


@dataclass
class TracedModel:
    gm: fx.GraphModule
    input_shape: tuple[int, ...]
    _out_shapes: dict[str, tuple[int, ...]] | None = field(default=None, repr=False)
    _in_shapes: dict[str, tuple[int, ...]] | None = field(default=None, repr=False)
    _sequential_pairs: list[tuple[str, str]] | None = field(default=None, repr=False)
    _dependency_pairs: list[tuple[str, str]] | None = field(default=None, repr=False)
    _hidden_modules: list[str] | None = field(default=None, repr=False)
    _param_count: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.input_shape = tuple(int(x) for x in self.input_shape)

    @classmethod
    def create(
        cls,
        model: nn.Module | fx.GraphModule,
        input_shape: tuple[int, ...],
    ) -> TracedModel:
        """Trace *model* when needed and attach *input_shape* for ShapeProp."""
        gm = extract_graph(model)
        return cls(gm, input_shape)

    def probe(self) -> torch.Tensor:
        """Random probe tensor aligned to the graph device and dtype."""
        return LayerShapeAnalyser.make_probe(self.gm, self.input_shape)

    def update_shapes(self) -> None:
        """Run ShapeProp and refresh cached layer shape maps."""
        self._out_shapes, self._in_shapes = LayerShapeAnalyser.collect_layer_shapes(
            self.gm, self.probe()
        )

    def shapes(self) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        """Return (output_shapes, input_shapes), computing once when needed."""
        if self._out_shapes is None or self._in_shapes is None:
            self.update_shapes()
        return self._out_shapes, self._in_shapes

    def sequential_pairs(self) -> list[tuple[str, str]]:
        if self._sequential_pairs is None:
            self._sequential_pairs = GraphStructureQuery.module_sequential_pairs(self.gm)
        return self._sequential_pairs

    def dependency_pairs(self) -> list[tuple[str, str]]:
        if self._dependency_pairs is None:
            self._dependency_pairs = GraphStructureQuery.module_dependency_pairs(self.gm)
        return self._dependency_pairs

    def hidden_modules(self) -> list[str]:
        if self._hidden_modules is None:
            self._hidden_modules = GraphStructureQuery.get_all_hidden_modules(self.gm)
        return self._hidden_modules

    def param_count(self) -> int:
        if self._param_count is None:
            self._param_count = GraphStructureQuery.get_amount_of_parameters(self.gm)
        return self._param_count

    def invalidate(self) -> None:
        """Clear cached analysis after graph or width mutations."""
        self._out_shapes = None
        self._in_shapes = None
        self._sequential_pairs = None
        self._dependency_pairs = None
        self._hidden_modules = None
        self._param_count = None
