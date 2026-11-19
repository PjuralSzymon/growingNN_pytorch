"""Conservative width resizing for functional FX Linear chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
import operator

from growingnn.utils.fx.operation_descriptors import (
    LinearOperationDescriptor,
    _resolve_attr,
    describe_linear_operations,
)
from growingnn.utils.quaziIdentity import get_reshsper


@dataclass(frozen=True)
class FunctionalLinearResizePlan:
    """One or more functional projections plus a compatible downstream consumer."""

    source: LinearOperationDescriptor
    sink: LinearOperationDescriptor
    pointwise_nodes: tuple[fx.Node, ...]
    old_width: int
    new_width: int
    cooperating_sources: tuple[LinearOperationDescriptor, ...] = ()

    @property
    def sources(self) -> tuple[LinearOperationDescriptor, ...]:
        return (self.source, *self.cooperating_sources)

    @property
    def added_parameter_count(self) -> int:
        delta = self.new_width - self.old_width
        source_added = sum(delta * (source.in_features or 0) for source in self.sources)
        source_added += sum(delta for source in self.sources if source.bias_node is not None)
        sink_added = delta * (self.sink.out_features or 0)
        return source_added + sink_added


def _parameter_aliases(
    gm: fx.GraphModule,
    parameter: nn.Parameter,
    alias_index: dict[int, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    if alias_index is not None:
        return alias_index.get(id(parameter), ())
    try:
        return tuple(
            name for name, item in gm.named_parameters(remove_duplicate=False)
            if item is parameter
        )
    except TypeError:
        return tuple(name for name, item in gm.named_parameters() if item is parameter)


def _parameter_for_node(gm: fx.GraphModule, node: fx.Node | None) -> nn.Parameter | None:
    if node is None or node.op != "get_attr":
        return None
    value: Any = _resolve_attr(gm, str(node.target))
    return value if isinstance(value, nn.Parameter) else None


def _is_safe_unary_feature_op(gm: fx.GraphModule, node: fx.Node, predecessor: fx.Node) -> bool:
    if node.all_input_nodes != [predecessor]:
        return False
    if node.op == "call_module":
        try:
            module = gm.get_submodule(str(node.target))
        except AttributeError:
            return False
        return isinstance(
            module,
            (nn.Identity, nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.Dropout),
        )
    if node.op == "call_method":
        name = str(node.target).lower()
    elif node.op == "call_function":
        target = node.target
        name = getattr(target, "__name__", "").lower()
        target_name = str(target).lower()
        if target_name.startswith("aten."):
            aten_name = target_name.split(".", 2)[1]
            return aten_name in {
                "relu", "gelu", "silu", "tanh", "sigmoid", "dropout", "clone", "contiguous"
            }
    else:
        return False
    return name in {
        "identity", "relu", "gelu", "silu", "tanh", "sigmoid", "dropout",
        "clone", "contiguous",
    }


def _is_tensor_multiply(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in (operator.mul, torch.mul):
        return len(node.all_input_nodes) == 2
    target = str(node.target).lower()
    return target.startswith(("aten.mul.tensor", "aten.mul.default")) and len(node.all_input_nodes) == 2


def _tensor_signature(node: fx.Node) -> tuple[tuple[int, ...], str | None, str | None] | None:
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("tensor_meta")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError, RuntimeError):
        return None
    return (
        dims,
        None if getattr(value, "dtype", None) is None else str(value.dtype),
        None if getattr(value, "device", None) is None else str(value.device),
    )


def _trace_back_to_functional_linear(
    gm: fx.GraphModule,
    node: fx.Node,
    by_node: dict[fx.Node, LinearOperationDescriptor],
) -> tuple[LinearOperationDescriptor | None, tuple[fx.Node, ...], str | None]:
    """Follow a unique unary feature-preserving branch back to a projection."""
    current = node
    reverse_ops: list[fx.Node] = []
    while current not in by_node:
        inputs = list(current.all_input_nodes)
        if len(inputs) != 1:
            return None, (), f"gated multiply sibling {current.name} is not a unary path"
        predecessor = inputs[0]
        if not _is_safe_unary_feature_op(gm, current, predecessor):
            return None, (), (
                f"gated multiply sibling operation {current.name} ({current.target!s}) "
                "is not a supported feature-preserving unary operation"
            )
        if len(predecessor.users) != 1:
            return None, (), f"gated multiply sibling path at {predecessor.name} has fanout"
        reverse_ops.append(current)
        current = predecessor
    descriptor = by_node[current]
    if descriptor.form not in ("function", "addmm") or not descriptor.recognized:
        return None, (), f"gated multiply sibling {current.name} is not a supported functional projection"
    if len(descriptor.node.users) != 1:
        return None, (), (
            f"gated multiply sibling projection {descriptor.node.name} has "
            f"{len(descriptor.node.users)} consumers; fanout resizing is unsupported"
        )
    return descriptor, tuple(reversed(reverse_ops)), None


class FunctionalLinearResizeAnalyzer:
    """Reusable per-graph index for checking many functional resize candidates."""

    def __init__(self, gm: fx.GraphModule):
        self.gm = gm
        self.descriptors = tuple(describe_linear_operations(gm))
        self.by_node = {item.node: item for item in self.descriptors}
        self.by_name = {item.node.name: item for item in self.descriptors}
        aliases: dict[int, list[str]] = {}
        try:
            parameters = gm.named_parameters(remove_duplicate=False)
        except TypeError:
            parameters = gm.named_parameters()
        for name, parameter in parameters:
            aliases.setdefault(id(parameter), []).append(name)
        self.alias_index = {key: tuple(names) for key, names in aliases.items()}

    def plan(
        self, node: fx.Node | str, new_width: int
    ) -> tuple[FunctionalLinearResizePlan | None, str | None]:
        return functional_linear_resize_plan(
            self.gm, node, new_width, _analyzer=self
        )


def functional_linear_resize_plan(
    gm: fx.GraphModule,
    node: fx.Node | str,
    new_width: int,
    *,
    _analyzer: FunctionalLinearResizeAnalyzer | None = None,
) -> tuple[FunctionalLinearResizePlan | None, str | None]:
    """Analyze a functional Linear resize across a supported graph region.

    Supported forms are one projection through a single-use pointwise path to
    one functional Linear, or two same-input projections joined by a compatible
    tensor multiply and followed by one functional Linear. Every participating
    projection and boundary is checked; fanout, unrelated merges, layout
    changes, module sinks, tied weights, and unknown metadata are rejected with
    a specific reason instead of being treated as generic passthroughs.
    """
    analyzer = _analyzer or FunctionalLinearResizeAnalyzer(gm)
    by_node = analyzer.by_node
    if isinstance(node, str):
        descriptor = analyzer.by_name.get(node)
        node = descriptor.node if descriptor is not None else next(
            (item for item in gm.graph.nodes if item.name == node), None
        )
    if node is None:
        return None, "functional Linear FX node is not present in the graph"
    source = by_node.get(node)
    if source is None or source.form not in ("function", "addmm"):
        return None, f"node {node.name!r} is not a recognized functional projection"
    if not source.recognized:
        return None, source.diagnostic or "functional projection parameters could not be resolved"
    if source.in_features is None or source.out_features is None:
        return None, "functional projection input or output width is unknown"
    if new_width <= source.out_features:
        return None, f"requested width {new_width} must be larger than {source.out_features}"

    source_weight = _parameter_for_node(gm, source.weight_node)
    if source_weight is None:
        return None, "functional projection weight is not an FX get_attr Parameter"
    aliases = _parameter_aliases(gm, source_weight, analyzer.alias_index)
    if len(aliases) != 1:
        return None, f"functional projection weight is tied or aliased: {aliases}"
    group_aliases = set(aliases)
    if source.bias_node is not None:
        bias = _parameter_for_node(gm, source.bias_node)
        if bias is None:
            return None, "functional projection bias is not a Parameter"
        bias_aliases = _parameter_aliases(gm, bias, analyzer.alias_index)
        if len(bias_aliases) != 1:
            return None, f"functional projection bias is tied or aliased: {bias_aliases}"
        group_aliases.update(bias_aliases)

    value_meta = node.meta.get("val")
    if value_meta is None:
        value_meta = node.meta.get("tensor_meta")
    shape = getattr(value_meta, "shape", None)
    if shape is None:
        return None, f"functional projection {node.name} has no output shape metadata"
    try:
        output_shape = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError, RuntimeError):
        return None, f"functional projection {node.name} has symbolic or unreadable output shape"
    if not output_shape or output_shape[-1] != source.out_features:
        return None, (
            f"functional projection output metadata {output_shape} does not end in width "
            f"{source.out_features}"
        )

    pointwise_nodes: list[fx.Node] = []
    cooperating_sources: list[LinearOperationDescriptor] = []
    current = node
    while True:
        if len(current.users) != 1:
            return None, (
                f"functional projection path at {current.name} has {len(current.users)} consumers; "
                "fanout resizing is unsupported"
            )
        user = next(iter(current.users))
        sink = by_node.get(user)
        if sink is not None:
            if sink.form not in ("function", "addmm") or not sink.recognized:
                return None, f"downstream projection {user.name} is not a supported functional projection"
            if sink.input_node is not current:
                return None, f"downstream projection {user.name} does not consume the resized feature directly"
            if sink.in_features != source.out_features:
                return None, (
                    f"downstream projection {user.name} expects {sink.in_features} features, "
                    f"but producer outputs {source.out_features}"
                )
            sink_weight = _parameter_for_node(gm, sink.weight_node)
            if sink_weight is None:
                return None, f"downstream projection {user.name} weight is not a Parameter"
            sink_aliases = _parameter_aliases(gm, sink_weight, analyzer.alias_index)
            if len(sink_aliases) != 1:
                return None, f"downstream projection weight is tied or aliased: {sink_aliases}"
            if group_aliases & set(sink_aliases):
                return None, "producer and consumer projections share a tied weight"
            if sink.bias_node is not None:
                sink_bias = _parameter_for_node(gm, sink.bias_node)
                if sink_bias is None:
                    return None, f"downstream projection {user.name} bias is not a Parameter"
                sink_bias_aliases = _parameter_aliases(gm, sink_bias, analyzer.alias_index)
                if len(sink_bias_aliases) != 1:
                    return None, f"downstream projection bias is tied or aliased: {sink_bias_aliases}"
                if group_aliases & set(sink_bias_aliases):
                    return None, "producer and consumer projections share a tied parameter"
            return FunctionalLinearResizePlan(
                source,
                sink,
                tuple(pointwise_nodes),
                source.out_features,
                new_width,
                tuple(cooperating_sources),
            ), None
        if user.op == "output":
            return None, "functional projection reaches graph output without an adjustable downstream Linear"
        if _is_tensor_multiply(user):
            inputs = list(user.all_input_nodes)
            siblings = [value for value in inputs if value is not current]
            if len(siblings) != 1:
                return None, f"gated multiply {user.name} does not have exactly two distinct tensor inputs"
            sibling_source, sibling_ops, sibling_reason = _trace_back_to_functional_linear(
                gm, siblings[0], by_node
            )
            if sibling_source is None:
                return None, sibling_reason or "gated multiply sibling projection is unsupported"
            if sibling_source.node is source.node or sibling_source in cooperating_sources:
                return None, f"gated multiply {user.name} reuses the same projection branch"
            if sibling_source.input_node is not source.input_node:
                return None, (
                    f"gated multiply projections {source.node.name} and {sibling_source.node.name} "
                    "do not share the same activation input"
                )
            if sibling_source.out_features != source.out_features:
                return None, (
                    f"gated multiply projections have different widths "
                    f"({source.out_features} and {sibling_source.out_features})"
                )
            main_signature = _tensor_signature(source.node)
            sibling_signature = _tensor_signature(sibling_source.node)
            merge_signature = _tensor_signature(user)
            if main_signature is None or sibling_signature is None or merge_signature is None:
                return None, f"gated multiply {user.name} or its branches have missing tensor metadata"
            if main_signature != sibling_signature or main_signature != merge_signature:
                return None, f"gated multiply {user.name} branch tensor interfaces do not match"

            sibling_weight = _parameter_for_node(gm, sibling_source.weight_node)
            if sibling_weight is None:
                return None, f"gated branch {sibling_source.node.name} weight is not a Parameter"
            sibling_weight_aliases = _parameter_aliases(
                gm, sibling_weight, analyzer.alias_index
            )
            if len(sibling_weight_aliases) != 1:
                return None, f"gated branch weight is tied or aliased: {sibling_weight_aliases}"
            if group_aliases & set(sibling_weight_aliases):
                return None, "gated projection branches share a tied weight"
            sibling_aliases = set(sibling_weight_aliases)
            if sibling_source.bias_node is not None:
                sibling_bias = _parameter_for_node(gm, sibling_source.bias_node)
                if sibling_bias is None:
                    return None, f"gated branch {sibling_source.node.name} bias is not a Parameter"
                sibling_bias_aliases = _parameter_aliases(
                    gm, sibling_bias, analyzer.alias_index
                )
                if len(sibling_bias_aliases) != 1:
                    return None, f"gated branch bias is tied or aliased: {sibling_bias_aliases}"
                if group_aliases & set(sibling_bias_aliases):
                    return None, "gated projection branches share a tied parameter"
                sibling_aliases.update(sibling_bias_aliases)
            group_aliases.update(sibling_aliases)
            cooperating_sources.append(sibling_source)
            pointwise_nodes.extend(sibling_ops)
            pointwise_nodes.append(user)
            current = user
            continue
        if not _is_safe_unary_feature_op(gm, user, current):
            return None, (
                f"operation {user.name} ({user.target!s}) is not a supported single-input "
                "feature-preserving operation; merges and layout changes require coordinated resizing"
            )
        user_meta = user.meta.get("val")
        if user_meta is None:
            user_meta = user.meta.get("tensor_meta")
        user_shape = getattr(user_meta, "shape", None)
        if user_shape is None:
            return None, f"operation {user.name} has no tensor shape metadata"
        try:
            user_shape = tuple(int(dim) for dim in user_shape)
        except (TypeError, ValueError, RuntimeError):
            return None, f"operation {user.name} has symbolic or unreadable tensor shape metadata"
        if not user_shape or user_shape[-1] != source.out_features:
            return None, (
                f"operation {user.name} output shape {user_shape} does not preserve final feature width "
                f"{source.out_features}"
            )
        pointwise_nodes.append(user)
        current = user


def _set_attr(gm: fx.GraphModule, target: str, value: nn.Parameter) -> None:
    components = target.split(".")
    parent: Any = gm
    for component in components[:-1]:
        parent = getattr(parent, component)
    setattr(parent, components[-1], value)


def resize_functional_linear_output(
    gm: fx.GraphModule,
    node: fx.Node | str,
    new_width: int,
) -> fx.GraphModule:
    """Apply a validated functional Linear producer/consumer width resize."""
    plan, reason = functional_linear_resize_plan(gm, node, new_width)
    if plan is None:
        raise ValueError(reason or "functional Linear resize is unsupported")
    sink = plan.sink
    sink_weight = _parameter_for_node(gm, sink.weight_node)
    assert sink_weight is not None
    sink_in_map = get_reshsper(
        plan.old_width, plan.new_width,
        dtype=sink_weight.dtype, device=sink_weight.device,
    )
    replacements: list[tuple[str, nn.Parameter]] = []
    with torch.no_grad():
        # F.linear stores weights as (out, in), while direct addmm stores its
        # matrix as (in, out). A traced transpose retains the former orientation.
        if sink.weight_orientation == "in_features_by_out_features":
            transformed_sink_weight = sink_in_map.T @ sink_weight
        else:
            transformed_sink_weight = sink_weight @ sink_in_map
        new_sink_weight = nn.Parameter(
            transformed_sink_weight.contiguous(),
            requires_grad=sink_weight.requires_grad,
        )
        replacements.append((str(sink.weight_target), new_sink_weight))
        for source in plan.sources:
            source_weight = _parameter_for_node(gm, source.weight_node)
            assert source_weight is not None
            source_out_map = get_reshsper(
                plan.old_width, plan.new_width,
                dtype=source_weight.dtype, device=source_weight.device,
            )
            if source.weight_orientation == "in_features_by_out_features":
                transformed_source_weight = source_weight @ source_out_map
            else:
                transformed_source_weight = source_out_map.T @ source_weight
            replacements.append((
                str(source.weight_target),
                nn.Parameter(
                    transformed_source_weight.contiguous(),
                    requires_grad=source_weight.requires_grad,
                ),
            ))
            if source.bias_node is not None:
                source_bias = _parameter_for_node(gm, source.bias_node)
                assert source_bias is not None
                replacements.append((
                    str(source.bias_node.target),
                    nn.Parameter(
                        (source_out_map.T @ source_bias).contiguous(),
                        requires_grad=source_bias.requires_grad,
                    ),
                ))

    replaced_targets: set[str] = set()
    for target, parameter in replacements:
        _set_attr(gm, target, parameter)
        replaced_targets.add(target)

    # Parameter get_attr nodes feed transpose/cast/projection nodes and may have
    # users well beyond the local producer/consumer pair. Clear annotations over
    # the full affected dataflow so later planning never treats old dimensions as
    # current metadata. Unrelated branches retain their cached annotations.
    affected = {
        node for node in gm.graph.nodes
        if node.op == "get_attr" and str(node.target) in replaced_targets
    }
    pending = list(affected)
    while pending:
        node = pending.pop()
        for user in node.users:
            if user not in affected:
                affected.add(user)
                pending.append(user)
    for node in affected:
        node.meta.pop("val", None)
        node.meta.pop("tensor_meta", None)
    gm.graph.lint()
    gm.recompile()
    return gm
