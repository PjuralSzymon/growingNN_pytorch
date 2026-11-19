"""Structural single-entry/single-exit regions over an FX dataflow graph.

Regions intentionally describe graph nodes and crossing edges independently of
module names and tensor metadata. They are analysis results, not mutation plans.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.fx as fx

from growingnn.utils.fx.operation_descriptors import is_functional_convolution_operation


@dataclass(frozen=True)
class BoundaryEdge:
    """An edge crossing a region boundary (the outside endpoint may be graph I/O)."""

    source: fx.Node | None
    destination: fx.Node | None
    kind: str
    # FX user maps identify a node pair, not an individual operand occurrence.
    # Keep the paths so repeated uses (for example add(x, x)) remain explicit.
    use_paths: tuple[tuple[Any, ...], ...] = ()


@dataclass(frozen=True)
class ValueInterface:
    """Tensor or structured value metadata at one crossing edge."""

    kind: str
    shape: tuple[Any, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    layout: str | None = None
    stride: tuple[Any, ...] | None = None
    requires_grad: bool | None = None
    children: tuple["ValueInterface", ...] = ()
    keys: tuple[Any, ...] = ()
    type_name: str | None = None
    constant: Any | None = None


@dataclass(frozen=True)
class FeatureTensorInterface:
    """Resolved tensor feature axis for a transformation that operates on features."""

    value: ValueInterface
    feature_axis: int | None
    features: int | None
    dtype: torch.dtype | None
    device: torch.device | None
    rejection_reason: str | None = None

    @property
    def eligible(self) -> bool:
        return self.rejection_reason is None


@dataclass(frozen=True)
class PairTraversalDecision:
    """Bounded-work decision for legacy exhaustive module-pair searches."""

    use_legacy_pairs: bool
    graph_nodes: int
    module_calls: int
    estimated_walk_steps: int
    reasons: tuple[str, ...] = ()


def pair_traversal_decision(
    gm: fx.GraphModule,
    *,
    max_estimated_walk_steps: int = 50_000,
    max_functional_to_module_ratio: float = 2.0,
) -> PairTraversalDecision:
    """Choose between legacy module-pair scans and bounded FX-region sites.

    Pair queries start a reachability walk from each editable module. Their
    worst-case work therefore grows with both module count and graph size. When
    that estimate or the functional-operation ratio is high, callers should use
    region/edge action sites instead. This is routing policy only; public pair
    query APIs remain unchanged.
    """
    nodes = list(gm.graph.nodes)
    module_calls = sum(node.op == "call_module" for node in nodes)
    functional_ops = sum(node.op in ("call_function", "call_method") for node in nodes)
    estimated = len(nodes) * module_calls
    reasons: list[str] = []
    if estimated > max_estimated_walk_steps:
        reasons.append(
            f"estimated module-pair reachability work {estimated} exceeds budget {max_estimated_walk_steps}"
        )
    if functional_ops > max(32, max_functional_to_module_ratio * module_calls):
        reasons.append(
            f"functional operation count {functional_ops} is high relative to {module_calls} module calls"
        )
    return PairTraversalDecision(
        use_legacy_pairs=not reasons,
        graph_nodes=len(nodes),
        module_calls=module_calls,
        estimated_walk_steps=estimated,
        reasons=tuple(reasons),
    )


@dataclass
class RegionCandidate:
    """A node-defined region with explicit boundary and analysis diagnostics."""

    entry: fx.Node
    exit: fx.Node
    nodes: tuple[fx.Node, ...]
    inputs: tuple[BoundaryEdge, ...]
    outputs: tuple[BoundaryEdge, ...]
    structural_reasons: tuple[str, ...] = ()
    interface_reasons: tuple[str, ...] = ()
    input_interface: tuple[ValueInterface, ...] = ()
    output_interface: tuple[ValueInterface, ...] = ()
    module_paths: tuple[str, ...] = ()
    # Empty for the root graph; nested graph module path for e.g. torch.cond arms.
    graph_path: tuple[str, ...] = ()

    @property
    def structurally_valid(self) -> bool:
        return not self.structural_reasons

    @property
    def interface_analyzable(self) -> bool:
        return not self.interface_reasons

    @property
    def safe_interface(self) -> bool:
        """Conservative representability check; this does not prove mutation safety."""
        return self.structurally_valid and self.interface_analyzable

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)


@dataclass(frozen=True)
class RegionActionSite:
    """One output-boundary edge that an action could transform, with diagnostics."""

    region: RegionCandidate
    edge: BoundaryEdge
    interface: ValueInterface
    rejection_reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return not self.rejection_reasons

    @property
    def meaningful_seam(self) -> bool:
        """Whether this edge is an architectural boundary worth considering.

        This is a topology filter, not an interface-safety check. It selects
        module boundaries and graph fan-out/merge points while avoiding an
        action proposal on every edge in a long functional chain. Consumers
        must still check ``eligible`` and their own operation semantics.
        """
        source, destination = self.edge.source, self.edge.destination
        if source is None or destination is None or destination.op == "output":
            return False
        return (
            source.op == "call_module"
            or destination.op == "call_module"
            or is_functional_convolution_operation(source)
            or is_functional_convolution_operation(destination)
            or len(source.users) > 1
            or len(source.all_input_nodes) > 1
        )


def _predecessors(node: fx.Node) -> set[fx.Node]:
    return set(node.all_input_nodes)


def _successors(node: fx.Node) -> set[fx.Node]:
    return set(node.users)


def _node_use_paths(value: Any, source: fx.Node, prefix: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """Find every leaf path to ``source`` in an FX argument pytree."""
    if value is source:
        return [prefix]
    paths: list[tuple[Any, ...]] = []
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            paths.extend(_node_use_paths(child, source, (*prefix, index)))
    elif isinstance(value, slice):
        for field in ("start", "stop", "step"):
            paths.extend(_node_use_paths(getattr(value, field), source, (*prefix, field)))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(_node_use_paths(child, source, (*prefix, key)))
    return paths


def _edge_use_paths(source: fx.Node, destination: fx.Node) -> tuple[tuple[Any, ...], ...]:
    paths = _node_use_paths(destination.args, source, ("args",))
    paths.extend(_node_use_paths(destination.kwargs, source, ("kwargs",)))
    return tuple(paths)


def _dominator_tree(
    nodes: list[fx.Node], roots: set[fx.Node], *, reverse: bool = False
) -> tuple[dict[fx.Node, fx.Node | None], dict[fx.Node, tuple[int, int]]]:
    """Compute a dominator forest for an acyclic graph without quadratic bitsets.

    The immediate dominator of a DAG node is the lowest common ancestor of its
    predecessors in the already-built dominator tree. Reversing edges computes
    postdominators. Euler intervals then answer dominance queries in constant time.
    ``None`` acts as a virtual root when paths originate at multiple graph inputs.
    """
    ordered = list(reversed(nodes)) if reverse else list(nodes)
    rank = {node: index for index, node in enumerate(ordered)}
    parent: dict[fx.Node, fx.Node | None] = {}

    def intersect(left: fx.Node | None, right: fx.Node | None) -> fx.Node | None:
        while left is not right:
            if left is None or right is None:
                return None
            while rank[left] > rank[right]:
                left = parent[left]
                if left is None:
                    return None
            while rank[right] > rank[left]:
                right = parent[right]
                if right is None:
                    return None
        return left

    adjacent = _successors if reverse else _predecessors
    node_set = set(nodes)
    for node in ordered:
        if node in roots:
            parent[node] = None
            continue
        predecessors = [p for p in adjacent(node) if p in node_set and p in parent]
        if not predecessors:
            parent[node] = None
            continue
        common = predecessors[0]
        for predecessor in predecessors[1:]:
            common = intersect(common, predecessor)
        parent[node] = common

    children: dict[fx.Node | None, list[fx.Node]] = {None: []}
    for node, ancestor in parent.items():
        children.setdefault(ancestor, []).append(node)
        children.setdefault(node, [])
    intervals: dict[fx.Node, tuple[int, int]] = {}
    starts: dict[fx.Node, int] = {}
    cursor = 0
    for root in children[None]:
        stack = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                intervals[node] = (starts[node], cursor)
                continue
            starts[node] = cursor
            cursor += 1
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(children[node]))
    return parent, intervals


def _dominates(intervals: dict[fx.Node, tuple[int, int]], ancestor: fx.Node, node: fx.Node) -> bool:
    left = intervals.get(ancestor)
    right = intervals.get(node)
    return left is not None and right is not None and left[0] <= right[0] < left[1]


def _common_tree_ancestor(
    parent: dict[fx.Node, fx.Node | None], nodes: list[fx.Node]
) -> fx.Node | None:
    """Return the lowest shared ancestor for nodes in a dominator forest."""
    if not nodes:
        return None
    common: fx.Node | None = nodes[0]
    for node in nodes[1:]:
        ancestors: set[fx.Node] = set()
        current = common
        while current is not None:
            ancestors.add(current)
            current = parent.get(current)
        current = node
        while current is not None and current not in ancestors:
            current = parent.get(current)
        common = current
        if common is None:
            return None
    return common


def _nearest_postdominating_boundaries(
    nodes: list[fx.Node], parent: dict[fx.Node, fx.Node | None]
) -> tuple[dict[fx.Node, fx.Node | None], dict[fx.Node, tuple[fx.Node, ...]]]:
    """Summarize nearest module and merge boundaries in one reverse graph walk.

    Walking the postdominator parent chain separately for every entry can become
    quadratic on deep exported graphs with many module boundaries. The parent of
    a postdominator is later in FX topological order, so summaries can be reused
    from that parent in a single reverse pass.
    """
    nearest_module: dict[fx.Node, fx.Node | None] = {}
    nearest_merges: dict[fx.Node, tuple[fx.Node, ...]] = {}
    for node in reversed(nodes):
        ancestor = parent.get(node)
        if ancestor is None:
            nearest_module[node] = None
            nearest_merges[node] = ()
            continue
        nearest_module[node] = (
            ancestor
            if ancestor.op == "call_module"
            else nearest_module.get(ancestor)
        )
        following_merges = nearest_merges.get(ancestor, ())
        nearest_merges[node] = (
            ((ancestor,) if len(ancestor.all_input_nodes) > 1 else ())
            + following_merges
        )[:2]
    return nearest_module, nearest_merges


def _describe_value(meta: Any) -> ValueInterface:
    """Preserve tensor and pytree structure in the interface description."""
    if meta is None:
        return ValueInterface(kind="none", type_name="NoneType")
    if isinstance(meta, (bool, int, float, str)):
        return ValueInterface(kind="scalar", type_name=type(meta).__name__, constant=meta)
    if isinstance(meta, Mapping):
        return ValueInterface(
            kind="dict" if isinstance(meta, dict) else "mapping",
            keys=tuple(meta.keys()),
            children=tuple(_describe_value(value) for value in meta.values()),
            type_name=type(meta).__qualname__,
        )
    if meta is not None and (hasattr(meta, "shape") or hasattr(meta, "dtype")):
        shape = getattr(meta, "shape", None)
        stride = getattr(meta, "stride", None)
        if callable(stride):
            try:
                stride = stride()
            except Exception:
                stride = None
        try:
            stride = None if stride is None else tuple(stride)
        except TypeError:
            stride = None
        return ValueInterface(
            kind="tensor",
            shape=None if shape is None else tuple(shape),
            dtype=None if getattr(meta, "dtype", None) is None else str(meta.dtype),
            device=None if getattr(meta, "device", None) is None else str(meta.device),
            layout=None if getattr(meta, "layout", None) is None else str(meta.layout),
            stride=stride,
            requires_grad=getattr(meta, "requires_grad", None),
        )
    if isinstance(meta, (tuple, list)):
        return ValueInterface(
            kind="tuple" if isinstance(meta, tuple) else "list",
            children=tuple(_describe_value(value) for value in meta),
            type_name=type(meta).__qualname__,
        )
    return ValueInterface(kind="unsupported", type_name=type(meta).__name__)


def _unsupported_value_paths(value: ValueInterface, path: str = "") -> list[str]:
    """Describe unsupported leaves nested inside structured tensor interfaces."""
    if value.kind == "unsupported":
        return [f"{path or '<value>'} has unsupported type {value.type_name}"]
    if value.kind in ("dict", "mapping"):
        return [
            reason
            for key, child in zip(value.keys, value.children)
            for reason in _unsupported_value_paths(child, f"{path}[{key!r}]")
        ]
    if value.kind in ("tuple", "list"):
        return [
            reason
            for index, child in enumerate(value.children)
            for reason in _unsupported_value_paths(child, f"{path}[{index}]")
        ]
    return []


def _node_value_metadata(node: fx.Node) -> Any:
    """Prefer route-specific fake values, falling back to ShapeProp tensor metadata."""
    value = node.meta.get("val")
    return value if value is not None else node.meta.get("tensor_meta")


def _describe_node_value(gm: fx.GraphModule, node: fx.Node) -> ValueInterface:
    """Describe a node boundary without treating absent metadata as a real None.

    A real Python ``None`` may be a valid structured output leaf. In contrast,
    missing FX metadata means the interface has not been analyzed, which must
    remain visible to action planners as a diagnostic.
    """
    metadata = _node_value_metadata(node)
    if metadata is None and node.op == "get_attr":
        value: Any = gm
        try:
            for component in str(node.target).split("."):
                value = getattr(value, component)
        except AttributeError:
            value = None
        metadata = value
    if metadata is None:
        return ValueInterface(kind="unknown", type_name="missing FX value metadata")
    return _describe_value(metadata)


def _has_tensor_value(value: ValueInterface) -> bool:
    if value.kind == "tensor":
        return True
    return any(_has_tensor_value(child) for child in value.children)


def _participates_in_tensor_flow(node: fx.Node) -> bool:
    """Exclude known scalar/shape-only export nodes from activation dominance."""
    metadata = _node_value_metadata(node)
    if metadata is None:
        # Symbolic FX often has no value metadata until ShapeProp is requested.
        # Keep those nodes in the structural graph rather than guessing.
        return True
    return _has_tensor_value(_describe_value(metadata))


def _boundary_edge_kind(source: fx.Node) -> str:
    if source.op == "get_attr":
        return "parameter_or_buffer"
    target = str(source.target).lower()
    if source.op in ("call_method", "call_function") and any(
        name in target
        for name in ("size", "sym_size", "shape", "stride", "storage_offset", "numel", "dim", "dtype", "device")
    ):
        return "metadata"
    return "data"


def analyze_feature_tensor(
    value: ValueInterface,
    *,
    feature_axis: int | Mapping[int, int] = -1,
    supported_ranks: tuple[int, ...] = (2, 3),
) -> FeatureTensorInterface:
    """Resolve a supported feature dimension and tensor type with diagnostics.

    Callers may provide architecture/layout-specific axes and ranks. The default
    policy intentionally covers rank-2 matrices and rank-3 batch/sequence/tensor
    values, where the final axis is commonly the feature dimension.
    """
    if value.kind != "tensor":
        return FeatureTensorInterface(value, None, None, None, None, f"value kind {value.kind!r} is not a tensor")
    if value.shape is None:
        return FeatureTensorInterface(value, None, None, None, None, "tensor shape metadata is missing")
    rank = len(value.shape)
    if isinstance(feature_axis, Mapping):
        if rank not in feature_axis:
            return FeatureTensorInterface(
                value, None, None, None, None,
                f"rank {rank} has no configured feature axis (configured ranks: {tuple(feature_axis)})",
            )
        axis_policy = feature_axis[rank]
    else:
        axis_policy = feature_axis
    allowed_ranks = tuple(feature_axis) if isinstance(feature_axis, Mapping) else supported_ranks
    if rank not in allowed_ranks:
        return FeatureTensorInterface(
            value, None, None, None, None,
            f"rank {rank} has no configured feature-axis policy (supported ranks: {allowed_ranks})",
        )
    axis = axis_policy if axis_policy >= 0 else rank + axis_policy
    if axis < 0 or axis >= rank:
        return FeatureTensorInterface(value, None, None, None, None, f"feature axis {axis_policy} is invalid for rank {rank}")
    width = value.shape[axis]
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        return FeatureTensorInterface(
            value, axis, None, None, None,
            f"feature dimension at axis {axis} is not a known positive integer: {width!r}",
        )
    dtype_name = (value.dtype or "").removeprefix("torch.")
    dtype = getattr(torch, dtype_name, None) if dtype_name else None
    if not isinstance(dtype, torch.dtype) or not (dtype.is_floating_point or dtype.is_complex):
        return FeatureTensorInterface(value, axis, width, None, None, f"dtype {value.dtype!r} is not a supported floating tensor type")
    try:
        device = torch.device(value.device) if value.device else None
    except (TypeError, RuntimeError) as exc:
        return FeatureTensorInterface(value, axis, width, dtype, None, f"device metadata {value.device!r} is invalid: {exc}")
    return FeatureTensorInterface(value, axis, width, dtype, device)


class GraphRegionAnalyzer:
    """Discover nested SESE candidates using dominance and postdominance.

    This operates on any ``fx.GraphModule`` (including modules produced by
    export or a custom tracer). Functional, method, get_attr, and module nodes
    are treated uniformly as computations.
    """

    @staticmethod
    def discover(
        gm: fx.GraphModule,
        *,
        include_rejected: bool = True,
    ) -> list[RegionCandidate]:
        nodes = list(gm.graph.nodes)
        if not nodes:
            return []
        # Parameter/buffer reads are explicit FX nodes, but they are external
        # values, not activation-flow control points. Including them as roots in
        # dominator analysis makes a projection with input + weight + bias appear
        # to have three unrelated paths, hiding all useful export regions.
        flow_nodes = [
            n for n in nodes
            if n.op != "get_attr" and _participates_in_tensor_flow(n)
        ]
        if not flow_nodes:
            flow_nodes = nodes
        placeholders = {n for n in flow_nodes if n.op == "placeholder"}
        outputs = {n for n in flow_nodes if n.op == "output"}
        dom_tree, dom_intervals = _dominator_tree(flow_nodes, placeholders or {flow_nodes[0]})
        postdom_tree, postdom_intervals = _dominator_tree(
            flow_nodes, outputs or {flow_nodes[-1]}, reverse=True
        )
        flow_node_set = set(flow_nodes)
        activation_reachable = set(placeholders)
        for node in nodes:
            if node not in flow_node_set or node in placeholders:
                continue
            if any(predecessor in activation_reachable for predecessor in node.all_input_nodes):
                activation_reachable.add(node)
        order = {node: i for i, node in enumerate(nodes)}
        candidates: list[RegionCandidate] = []
        computations = [
            node for node in nodes if node.op not in ("placeholder", "output", "get_attr")
        ]
        # Keep operation singletons, then add canonical nested cuts: each structural
        # entry is paired with its nearest postdominating module boundary and its
        # first two postdominating merges. This captures module/attention/feed-forward
        # regions without enumerating every pair in a long module chain.
        boundary_nodes = [
            node for node in computations
            if node.op == "call_module" or len(node.users) > 1 or len(node.all_input_nodes) > 1
        ]
        nearest_module, nearest_merges = _nearest_postdominating_boundaries(
            nodes, postdom_tree
        )
        pairs = {(node, node) for node in computations}
        stack_region_nodes: dict[
            tuple[fx.Node, fx.Node], list[tuple[str, frozenset[fx.Node]]]
        ] = {}
        for entry in boundary_nodes:
            for exit_node in (nearest_module[entry], *nearest_merges[entry]):
                if (
                    exit_node is not None
                    and exit_node is not entry
                    and order[entry] <= order[exit_node]
                    and _dominates(dom_intervals, entry, exit_node)
                    and _dominates(postdom_intervals, exit_node, entry)
                ):
                    pairs.add((entry, exit_node))
        # Export and some custom tracers preserve source-module ancestry on
        # nodes. Use that metadata as a generic hint for nested cuts (block,
        # attention, MLP, etc.); region membership and validity still come from
        # FX dominance/postdominance, never from module-name equality alone.
        stack_groups: dict[str, list[fx.Node]] = {}
        for node in computations:
            stack = node.meta.get("nn_module_stack")
            if not isinstance(stack, dict):
                continue
            for stack_entry in stack.values():
                if not isinstance(stack_entry, (tuple, list)) or not stack_entry:
                    continue
                path = str(stack_entry[0])
                if path:
                    stack_groups.setdefault(path, []).append(node)
        for path, group_nodes in stack_groups.items():
            if not path:
                continue
            unique = sorted(set(group_nodes), key=order.__getitem__)
            # Stack groups can contain guards/assertions before the first tensor
            # operation. Keep data-producing values for cuts, while preserving
            # structured tensor outputs such as attention tuples.
            value_nodes = [
                node for node in unique
                if node in activation_reachable
                and _describe_value(_node_value_metadata(node)).kind
                in ("tensor", "tuple", "list", "dict")
            ]
            if len(value_nodes) > 1:
                entry = _common_tree_ancestor(dom_tree, value_nodes)
                exit_node = _common_tree_ancestor(postdom_tree, value_nodes)
                if entry is not None and exit_node is not None:
                    pairs.add((entry, exit_node))
                # Preserve the traced module's exact operation group as
                # an additional candidate even when side inputs prevent a
                # single dominator pair. The candidate will carry explicit
                # crossing edges and structural diagnostics below.
                group_entry, group_exit = value_nodes[0], value_nodes[-1]
                key = (group_entry, group_exit)
                pairs.add(key)
                # Include any metadata/assert operations with this hierarchy as
                # internal nodes too; otherwise their intermediate values appear
                # as spurious exits from the captured module region.
                stack_region_nodes.setdefault(key, []).append((path, frozenset(unique)))
        ordered_pairs = sorted(pairs, key=lambda pair: (order[pair[0]], order[pair[1]]))
        candidate_specs: list[
            tuple[fx.Node, fx.Node, frozenset[fx.Node] | None, str | None]
        ] = [(entry, exit_node, None, None) for entry, exit_node in ordered_pairs]
        for (entry, exit_node), groups in stack_region_nodes.items():
            candidate_specs.extend(
                (entry, exit_node, members, path) for path, members in groups
            )
        seen_regions: dict[
            tuple[fx.Node, fx.Node, tuple[fx.Node, ...]], RegionCandidate
        ] = {}
        for entry, exit_node, stack_region, module_path in candidate_specs:
            if stack_region is None and (
                exit_node.op in ("placeholder", "output")
                or not _dominates(dom_intervals, entry, exit_node)
                or not _dominates(postdom_intervals, exit_node, entry)
            ):
                continue
            if stack_region is not None:
                selected = set(stack_region)
            elif entry is exit_node:
                selected = {entry}
            else:
                lo, hi = order[entry], order[exit_node]
                selected = {
                    node for node in nodes[lo : hi + 1]
                    if _dominates(dom_intervals, entry, node)
                    and _dominates(postdom_intervals, exit_node, node)
                }
            selected.update((entry, exit_node))
            ordered = tuple(n for n in nodes[order[entry] : order[exit_node] + 1] if n in selected)
            inputs: list[BoundaryEdge] = []
            outputs_edges: list[BoundaryEdge] = []
            for node in ordered:
                for pred in node.all_input_nodes:
                    if pred not in selected:
                        inputs.append(BoundaryEdge(
                            pred, node, _boundary_edge_kind(pred), _edge_use_paths(pred, node)
                        ))
                for user in node.users:
                    if user not in selected:
                        outputs_edges.append(BoundaryEdge(
                            node, user, _boundary_edge_kind(node), _edge_use_paths(node, user)
                        ))
            reasons: list[str] = []
            if stack_region is not None and (
                not _dominates(dom_intervals, entry, exit_node)
                or not _dominates(postdom_intervals, exit_node, entry)
            ):
                reasons.append(
                    "capture-route module grouping does not form a dominance-validated single-entry/single-exit region"
                )
            # Parameters and buffers can be read at several operations inside a
            # transformer block. They remain explicit interface inputs, but do
            # not create additional activation entry points for SESE analysis.
            activation_inputs = [
                edge for edge in inputs
                if edge.source is None or edge.source.op != "get_attr"
            ]
            if len({edge.destination for edge in activation_inputs}) > 1:
                reasons.append("external dependencies enter through multiple nodes; not single-entry")
            if len({edge.source for edge in outputs_edges}) > 1:
                reasons.append("region exits through multiple nodes; not single-exit")
            interface_reasons: list[str] = []
            in_meta = tuple(_describe_node_value(gm, edge.source) for edge in inputs)
            out_meta = tuple(_describe_node_value(gm, edge.source) for edge in outputs_edges)
            for edge, desc in zip(inputs, in_meta):
                if desc.kind == "unknown":
                    interface_reasons.append(
                        f"input {edge.source.name} cannot be analyzed: tensor/value metadata is missing"
                    )
                interface_reasons.extend(
                    f"input {edge.source.name} {reason}"
                    for reason in _unsupported_value_paths(desc)
                )
            for edge, desc in zip(outputs_edges, out_meta):
                if desc.kind == "unknown":
                    interface_reasons.append(
                        f"output {edge.source.name} cannot be analyzed: tensor/value metadata is missing"
                    )
                interface_reasons.extend(
                    f"output {edge.source.name} {reason}"
                    for reason in _unsupported_value_paths(desc)
                )
            candidate = RegionCandidate(
                entry, exit_node, ordered, tuple(inputs), tuple(outputs_edges),
                tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(interface_reasons)),
                in_meta, out_meta, (module_path,) if module_path is not None else (),
            )
            region_key = (entry, exit_node, ordered)
            existing = seen_regions.get(region_key)
            if existing is not None:
                if module_path is not None and module_path not in existing.module_paths:
                    existing.module_paths = (*existing.module_paths, module_path)
                continue
            seen_regions[region_key] = candidate
            if include_rejected or candidate.structurally_valid:
                candidates.append(candidate)
        candidates.sort(key=lambda c: (len(c.nodes), order[c.entry], order[c.exit]))
        return candidates

    @staticmethod
    def discover_nested(
        gm: fx.GraphModule,
        *,
        include_rejected: bool = True,
    ) -> list[RegionCandidate]:
        """Discover regions in the root graph and nested FX GraphModules.

        ``torch.export`` represents structured control flow such as ``torch.cond``
        with branch GraphModules stored as child modules and referenced by
        ``get_attr`` operands. Those branches are separate control-flow scopes,
        so their nodes must not be mixed into root-graph reachability or passed
        to root-graph editors. This API returns each candidate with ``graph_path``
        identifying its owning GraphModule; ``discover`` and action-site
        generation continue to analyze only the supplied root graph.
        """
        candidates: list[RegionCandidate] = []
        for path, module in gm.named_modules():
            if not isinstance(module, fx.GraphModule):
                continue
            graph_path = tuple(path.split(".")) if path else ()
            candidates.extend(
                replace(candidate, graph_path=graph_path)
                for candidate in GraphRegionAnalyzer.discover(
                    module, include_rejected=include_rejected
                )
            )
        return candidates

    @staticmethod
    def describe_node_value(node: fx.Node) -> ValueInterface:
        """Public metadata adapter shared by graph transformations."""
        return _describe_value(_node_value_metadata(node))

    @staticmethod
    def discover_action_sites(
        gm: fx.GraphModule,
        *,
        include_rejected: bool = True,
        minimum_rank: int = 2,
    ) -> list[RegionActionSite]:
        """Find tensor-valued region exits without assuming their FX node kind.

        These are analysis sites, not approved mutations. An action must still check
        its own tensor semantics and graph-rewrite constraints before applying.
        """
        sites: list[RegionActionSite] = []
        seen: set[tuple[fx.Node, fx.Node, fx.Node, fx.Node]] = set()
        for region in GraphRegionAnalyzer.discover(gm, include_rejected=True):
            region_reasons = list(region.structural_reasons) + list(region.interface_reasons)
            for edge in region.outputs:
                if edge.source is None or edge.destination is None:
                    continue
                key = (region.entry, region.exit, edge.source, edge.destination)
                if key in seen:
                    continue
                seen.add(key)
                interface = _describe_value(_node_value_metadata(edge.source))
                reasons = list(region_reasons)
                if edge.source.op == "get_attr":
                    reasons.append(
                        f"output {edge.source.name} is a parameter or buffer value, not an activation seam"
                    )
                target_name = str(edge.destination.target).lower()
                source_target_name = str(edge.source.target).lower()
                metadata_accessors = (
                    "size", "sym_size", "shape", "stride", "storage_offset", "numel",
                    "dim", "dtype", "device", "layout", "getattr",
                )
                if edge.source.op in ("call_method", "call_function") and any(
                    accessor in source_target_name for accessor in metadata_accessors
                ):
                    reasons.append(
                        f"source {edge.source.name} computes tensor metadata ({source_target_name}), not an activation tensor"
                    )
                if edge.destination.op in ("call_method", "call_function") and any(
                    accessor in target_name for accessor in metadata_accessors
                ):
                    reasons.append(
                        f"destination {edge.destination.name} only reads tensor metadata ({target_name})"
                    )
                if interface.kind != "tensor":
                    reasons.append(f"output {edge.source.name} is {interface.kind}, not one tensor")
                elif interface.shape is None:
                    reasons.append(f"output {edge.source.name} has no shape metadata")
                elif len(interface.shape) < minimum_rank:
                    reasons.append(
                        f"output {edge.source.name} rank {len(interface.shape)} is below required rank {minimum_rank}"
                    )
                if interface.kind == "tensor" and interface.dtype is not None:
                    if not interface.dtype.startswith(("torch.float", "torch.bfloat", "torch.complex")):
                        reasons.append(f"output {edge.source.name} dtype {interface.dtype} is not a floating tensor")
                elif interface.kind == "tensor":
                    reasons.append(f"output {edge.source.name} has no dtype metadata")
                site = RegionActionSite(region, edge, interface, tuple(dict.fromkeys(reasons)))
                if include_rejected or site.eligible:
                    sites.append(site)
        return sites

