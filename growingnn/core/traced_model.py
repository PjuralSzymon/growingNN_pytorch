"""Traced FX model plus cached analysis for training and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.fx as fx
import torch.nn as nn

from growingnn.utils.fx.graph_analysis import GraphStructureQuery, LayerShapeAnalyser
from growingnn.utils.fx.capture import capture_graph, order_example_inputs
from growingnn.utils.fx.regions import GraphRegionAnalyzer, RegionActionSite, RegionCandidate


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
    _regions: list[RegionCandidate] | None = field(default=None, repr=False)
    _nested_regions: list[RegionCandidate] | None = field(default=None, repr=False)
    _action_sites: list[RegionActionSite] | None = field(default=None, repr=False)
    _example_inputs: tuple[object, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.input_shape = tuple(int(x) for x in self.input_shape)

    @classmethod
    def create(
        cls,
        model: nn.Module | fx.GraphModule,
        input_shape: tuple[int, ...],
        *,
        capture: str = "symbolic",
        example_args: tuple[object, ...] | None = None,
        example_kwargs: dict[str, object] | None = None,
        leaf_module_types: tuple[type[nn.Module], ...] = (),
    ) -> TracedModel:
        """Capture *model* when needed and attach *input_shape* for ShapeProp."""
        gm = model if isinstance(model, fx.GraphModule) else capture_graph(
            model,
            route=capture,
            example_args=example_args,
            example_kwargs=example_kwargs,
            leaf_module_types=leaf_module_types,
        )
        traced = cls(gm, input_shape)
        if example_args is not None or example_kwargs is not None:
            traced._example_inputs = order_example_inputs(
                gm, example_args or (), example_kwargs
            )
        return traced

    def probe(self) -> torch.Tensor:
        """Random probe tensor aligned to the graph device and dtype."""
        return LayerShapeAnalyser.make_probe(self.gm, self.input_shape)

    def update_shapes(self) -> None:
        """Run ShapeProp and refresh cached layer shape maps."""
        if self._example_inputs is not None:
            self._out_shapes, self._in_shapes = LayerShapeAnalyser.collect_layer_shapes(
                self.gm, example_inputs=self._example_inputs
            )
        else:
            self._out_shapes, self._in_shapes = LayerShapeAnalyser.collect_layer_shapes(
                self.gm, self.probe()
            )
        # ShapeProp writes value metadata onto FX nodes. Any earlier region or
        # action-site result may have recorded missing-interface diagnostics.
        self._regions = None
        self._action_sites = None

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

    def regions(self, *, include_rejected: bool = True) -> list[RegionCandidate]:
        """Discover structural SESE regions over all FX operations, including functional nodes."""
        if self._regions is None:
            self._regions = GraphRegionAnalyzer.discover(self.gm, include_rejected=True)
        if include_rejected:
            return self._regions
        return [candidate for candidate in self._regions if candidate.structurally_valid]

    def nested_regions(self, *, include_rejected: bool = True) -> list[RegionCandidate]:
        """Discover root and branch-graph regions with explicit graph paths.

        Nested branch candidates are descriptive. Existing action generators
        operate on root-graph nodes unless they explicitly add a branch-aware
        editor, so this API does not merge nested nodes into ``action_sites``.
        """
        if self._nested_regions is None:
            self._nested_regions = GraphRegionAnalyzer.discover_nested(
                self.gm, include_rejected=True
            )
        if include_rejected:
            return self._nested_regions
        return [candidate for candidate in self._nested_regions if candidate.structurally_valid]

    def action_sites(self, *, include_rejected: bool = True) -> list[RegionActionSite]:
        """Return FX-node-boundary sites and interface diagnostics for action analysis."""
        if self._action_sites is None:
            self._action_sites = GraphRegionAnalyzer.discover_action_sites(
                self.gm, include_rejected=True
            )
        if include_rejected:
            return self._action_sites
        return [site for site in self._action_sites if site.eligible]

    def invalidate(self) -> None:
        """Clear cached analysis after graph or width mutations."""
        self._out_shapes = None
        self._in_shapes = None
        self._sequential_pairs = None
        self._dependency_pairs = None
        self._hidden_modules = None
        self._param_count = None
        self._regions = None
        self._nested_regions = None
        self._action_sites = None
