"""Graph-level edits: add/remove layers and rewire connections."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from growingnn.utils.fx.graph_analysis import (
    GraphStructureQuery,
    LayerShapeAnalyser,
    GraphConnectivity,
    LayerBridgeFinder,
)
from growingnn.utils.fx.node_analysis import ModuleResolver
from growingnn.utils.fx.node_editor import NodeEditor
from growingnn.utils.fx.sum_nodes import connect_residual_branch, is_merge_branch_layer, is_sum_node, remove_layer_from_sums


def _insert_call_module_after(gm, insert_after, module_name, module_input):
    with gm.graph.inserting_after(insert_after):
        return gm.graph.call_module(module_name, args=(module_input,))


def _append_batchnorm_and_relu(gm, conv_node, conv_module: nn.Conv2d, name: str):
    """Place BatchNorm and ReLU after a new conv so stacked inserts stay nonlinear."""
    bn_name = f"{name}_bn"
    bn = _align_new_module_to_graph(gm, nn.BatchNorm2d(conv_module.out_channels))
    gm.add_module(bn_name, bn)
    bn_node = _insert_call_module_after(gm, conv_node, bn_name, conv_node)
    with gm.graph.inserting_after(bn_node):
        return gm.graph.call_function(F.relu, args=(bn_node,))


def _align_new_module_to_graph(gm: fx.GraphModule, new_layer):
    """
    Move *new_layer* onto the GraphModule device/dtype before insertion.

    Action factories often build modules on CPU. Live and simulation models may
    already be on CUDA; ShapeProp then crashes with mixed CUDA inputs / CPU weights.
    """
    try:
        param = next(gm.parameters())
        return new_layer.to(device=param.device, dtype=param.dtype).train(gm.training)
    except StopIteration:
        return new_layer.train(gm.training)


def _align_new_module_to_node(gm: fx.GraphModule, new_layer: nn.Module, node: fx.Node) -> nn.Module:
    """Align an FX edit to the actual source tensor, not an unrelated graph parameter."""
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("tensor_meta")
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    try:
        parameter = next(gm.parameters())
    except StopIteration:
        parameter = None
    # ShapeProp's TensorMetadata has dtype but no device. In that case the graph
    # parameter supplies placement only; the edge tensor remains authoritative
    # for dtype. FakeTensor/export metadata provides both values directly.
    if device is None and parameter is not None:
        device = parameter.device
    if not isinstance(dtype, torch.dtype) and parameter is not None:
        dtype = parameter.dtype
    if device is not None or dtype is not None:
        try:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if isinstance(dtype, torch.dtype):
                kwargs["dtype"] = dtype
            return new_layer.to(**kwargs).train(gm.training)
        except (TypeError, RuntimeError, ValueError):
            # Retain the caller's module instead of inferring a cast from a
            # different endpoint when the framework cannot apply this placement.
            pass
    return new_layer.train(gm.training)


def _replace_at_path(value, path, old, new):
    """Replace one FX Node occurrence in nested args/kwargs, preserving containers."""
    if not path:
        if value is not old:
            raise ValueError("The selected FX operand path does not contain the requested source")
        return new
    key, *tail = path
    if isinstance(value, tuple):
        items = list(value)
        items[key] = _replace_at_path(items[key], tail, old, new)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)
    if isinstance(value, list):
        items = list(value)
        items[key] = _replace_at_path(items[key], tail, old, new)
        return items
    if isinstance(value, slice):
        if key not in ("start", "stop", "step"):
            raise ValueError(f"Invalid slice operand path component {key!r}")
        fields = {field: getattr(value, field) for field in ("start", "stop", "step")}
        fields[key] = _replace_at_path(fields[key], tail, old, new)
        return slice(fields["start"], fields["stop"], fields["step"])
    if isinstance(value, Mapping):
        replacement = _replace_at_path(value[key], tail, old, new)
        try:
            items = copy.copy(value)
            items[key] = replacement
            return items
        except (TypeError, AttributeError):
            items = dict(value)
            items[key] = replacement
            try:
                return type(value)(items)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot preserve FX argument mapping type {type(value).__qualname__}"
                ) from exc
    raise ValueError(f"Unsupported FX operand path component {key!r}")


def bypass_shapes_compatible(
    predecessor_output_shape: tuple[int, ...] | None,
    successor_input_shape: tuple[int, ...] | None,
) -> bool:
    """True when a predecessor output can feed a successor input without a bridge layer."""
    return (
        predecessor_output_shape is not None
        and successor_input_shape is not None
        and predecessor_output_shape == successor_input_shape
    )


def compute_bypass_matching(
    input_layers: list[str],
    output_layers: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    input_shapes: dict[str, tuple[int, ...]],
) -> dict[str, str] | None:
    """Map each successor to one compatible predecessor using the fewest pairwise skips."""
    if not input_layers or not output_layers:
        return None

    matching: dict[str, str] = {}
    used_inputs: set[str] = set()

    for out_id in output_layers:
        succ_in_shape = input_shapes.get(out_id)
        if succ_in_shape is None:
            return None
        candidates = [
            in_id for in_id in input_layers
            if bypass_shapes_compatible(output_shapes.get(in_id), succ_in_shape)
        ]
        if not candidates:
            return None
        picked = next((c for c in candidates if c not in used_inputs), candidates[0])
        matching[out_id] = picked
        used_inputs.add(picked)

    return matching


def branch_only_bypass_compatible(
    layer_node: fx.Node,
    input_shapes: dict[str, tuple[int, ...]],
) -> bool:
    """True when a layer without sequential successors can be skipped via one FX input."""
    inputs = list(layer_node.all_input_nodes)
    if len(inputs) != 1:
        return False
    replacement_shape = LayerShapeAnalyser.node_shape(inputs[0])
    if replacement_shape is None or not layer_node.users:
        return False
    for user in layer_node.users:
        if is_sum_node(user) or user.op != "call_module":
            return False
        if not bypass_shapes_compatible(replacement_shape, input_shapes.get(str(user.target))):
            return False
    return True


def _producer_before_layer(
    gm: fx.GraphModule,
    layer_node: fx.Node,
    input_layer_id: str,
) -> fx.Node:
    """Return the FX node that feeds layer_node on the path from input_layer_id."""
    src = ModuleResolver.find_call_module(gm.graph.nodes, input_layer_id)
    path = LayerBridgeFinder.path_dst_to_src(layer_node, src)
    if path is None:
        return src
    return path[1] if len(path) >= 2 else src


def _reachable_output_layers(
    start: fx.Node,
    output_layer_ids: set[str],
) -> list[str]:
    """Return output layer ids reachable forward from start without crossing another call_module target."""
    found: list[str] = []
    stack = [start]
    seen: set[fx.Node] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.op == "call_module" and str(node.target) in output_layer_ids:
            found.append(str(node.target))
            continue
        stack.extend(node.users)
    return list(dict.fromkeys(found))


def _rewire_branch_only_layer(layer_node: fx.Node) -> None:
    """Wire the single FX producer of a branch-only layer directly to its users."""
    inputs = list(layer_node.all_input_nodes)
    if len(inputs) != 1:
        raise ValueError("Branch-only layer delete requires exactly one FX input")
    replacement = inputs[0]
    for user in layer_node.users.copy():
        user.replace_input_with(layer_node, replacement)


def _rewire_layer_users(
    gm: fx.GraphModule,
    layer_node: fx.Node,
    matching: dict[str, str],
    output_layers: list[str],
) -> None:
    """Replace layer_node in each user with the minimal compatible predecessor branch."""
    if len(matching) == 1:
        replacement = _producer_before_layer(gm, layer_node, next(iter(matching.values())))
        for user in layer_node.users.copy():
            user.replace_input_with(layer_node, replacement)
        return

    default_out = output_layers[0]
    for user in layer_node.users.copy():
        reached = _reachable_output_layers(user, set(matching))
        out_id = reached[0] if reached else default_out
        replacement = _producer_before_layer(gm, layer_node, matching[out_id])
        user.replace_input_with(layer_node, replacement)


def prune_unreachable_nodes(gm: fx.GraphModule) -> list[str]:
    """Erase nodes not on any path to output and drop orphaned submodules."""
    live = GraphConnectivity.nodes_reachable_from_output(gm)
    removed_modules: list[str] = []
    for node in reversed(list(gm.graph.nodes)):
        if node in live:
            continue
        if node.op == "call_module":
            removed_modules.append(str(node.target))
        gm.graph.erase_node(node)
    live_targets = {str(node.target) for node in gm.graph.nodes if node.op == "call_module"}
    for name in dict.fromkeys(removed_modules):
        if hasattr(gm, name) and name not in live_targets:
            delattr(gm, name)
    if removed_modules:
        gm.graph.lint()
        gm.recompile()
    return list(dict.fromkeys(removed_modules))


class ModelStructureEditor:
    """Add and remove layers in an FX graph."""

    @staticmethod
    def insert_module_on_edge(
        gm: fx.GraphModule,
        source: fx.Node,
        destination: fx.Node,
        new_layer: nn.Module,
        name: str,
        *,
        use_path: tuple[object, ...] | None = None,
    ) -> fx.Node:
        """Insert a module on one dataflow edge, regardless of endpoint node kinds.

        The source or destination may be a ``call_module``, ``call_function``,
        ``call_method``, ``get_attr``, or graph output node. Interface compatibility
        remains the caller's responsibility and should be checked from node metadata.
        """
        if destination not in source.users:
            raise ValueError(
                f"Cannot insert on non-edge {source.name!r} -> {destination.name!r}"
            )
        from growingnn.utils.fx.regions import _edge_use_paths
        paths = _edge_use_paths(source, destination)
        if use_path is None:
            if len(paths) != 1:
                raise ValueError(
                    f"Edge {source.name!r} -> {destination.name!r} has {len(paths)} operand uses; "
                    "pass use_path to edit exactly one occurrence"
                )
            use_path = paths[0]
        elif use_path not in paths:
            raise ValueError("use_path is not an occurrence of the requested FX edge")
        if hasattr(gm, name):
            raise ValueError(f"GraphModule already has a child named {name!r}")
        new_layer = _align_new_module_to_node(gm, new_layer, source)
        original_args = destination.args
        original_kwargs = destination.kwargs
        gm.add_module(name, new_layer)
        inserted = None
        try:
            with gm.graph.inserting_before(destination):
                inserted = gm.graph.call_module(name, args=(source,))
            root_name, *path = use_path
            if root_name == "args":
                destination.args = _replace_at_path(destination.args, path, source, inserted)
            elif root_name == "kwargs":
                destination.kwargs = _replace_at_path(destination.kwargs, path, source, inserted)
            else:
                raise ValueError("FX operand path must start with 'args' or 'kwargs'")
            gm.graph.lint()
            gm.recompile()
        except Exception as exc:
            # Do not leave an orphan module or a half-rewired graph when a
            # structured argument container cannot be rebuilt or graph linting
            # rejects the edit.
            destination.args = original_args
            destination.kwargs = original_kwargs
            if inserted is not None and not inserted.users:
                gm.graph.erase_node(inserted)
            delattr(gm, name)
            try:
                gm.graph.lint()
                gm.recompile()
            except Exception as rollback_error:
                exc.add_note(f"FX edge-insertion rollback also failed: {rollback_error}")
            raise
        return inserted

    @staticmethod
    def add_residual_branch_on_nodes(
        gm: fx.GraphModule,
        destination: fx.Node,
        source: fx.Node,
        new_layer: nn.Module,
        name: str,
    ) -> None:
        """Add a transformed FX value as a residual term at a graph-node boundary."""
        if destination.graph is not gm.graph or source.graph is not gm.graph:
            raise ValueError("Residual source and destination must belong to the GraphModule")
        if hasattr(gm, name):
            raise ValueError(f"GraphModule already has a child named {name!r}")
        new_layer = _align_new_module_to_node(gm, new_layer, source)
        gm.add_module(name, new_layer)
        connect_residual_branch(gm, destination, source, name)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def add_new_residual_layer(gm, src_name, dst_name, new_layer, name):
        """Insert *new_layer* as a residual branch from *src_name* added to *dst_name* output."""
        new_layer = _align_new_module_to_graph(gm, new_layer)
        gm.add_module(name, new_layer)
        nodes = list(gm.graph.nodes)
        connect_residual_branch(
            gm,
            ModuleResolver.find_call_module(nodes, dst_name),
            ModuleResolver.find_call_module(nodes, src_name),
            name,
        )
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def add_new_seq_layer(gm, src_name, dst_name, new_layer, name, add_batchnorm_and_relu: bool = False):
        """Insert *new_layer* sequentially on the path from *src_name* to *dst_name*."""
        new_layer = _align_new_module_to_graph(gm, new_layer)
        gm.add_module(name, new_layer)
        nodes = list(gm.graph.nodes)
        src = ModuleResolver.find_call_module(nodes, src_name)
        dst = ModuleResolver.find_call_module(nodes, dst_name)
        if src is dst:
            raise ValueError("src and dst must differ.")

        path = LayerBridgeFinder.path_dst_to_src(dst, src)
        if path is None:
            raise ValueError(f"No path from {dst_name!r} back to {src_name!r} in the FX graph.")

        new_out = _insert_call_module_after(gm, path[1], name, path[1])
        tail = _append_batchnorm_and_relu(gm, new_out, new_layer, name) if add_batchnorm_and_relu else new_out
        NodeEditor.swap_node_input(dst, path[1], tail)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def add_new_seq_layer_before_flatten(
        gm, src_name, dst_name, new_layer, name,
        add_batchnorm_and_relu: bool = False,
        before_global_pool: bool = True,
    ):
        """Insert *new_layer* immediately before the flatten node on the src→dst sequential path."""
        from growingnn.utils.fx.graph_analysis import GraphStructureQuery

        new_layer = _align_new_module_to_graph(gm, new_layer)
        gm.add_module(name, new_layer)
        nodes = list(gm.graph.nodes)
        src = ModuleResolver.find_call_module(nodes, src_name)
        dst = ModuleResolver.find_call_module(nodes, dst_name)
        if src is dst:
            raise ValueError("src and dst must differ.")

        path_destination_to_source = LayerBridgeFinder.path_dst_to_src(dst, src)
        if path_destination_to_source is None:
            raise ValueError(f"No path from {dst_name!r} back to {src_name!r} in the FX graph.")

        flatten_node = GraphStructureQuery.find_flatten_node_on_path_toward_source(
            path_destination_to_source, gm,
        )
        if flatten_node is None:
            raise ValueError(f"No flatten on path from {src_name!r} to {dst_name!r}.")

        pools_before_flatten = GraphStructureQuery.find_pool_nodes_between_flatten_and_source(
            path_destination_to_source, flatten_node, gm,
        )
        if not pools_before_flatten:
            raise ValueError("Sequential conv before flatten requires at least one pool before flatten.")
        flatten_index = path_destination_to_source.index(flatten_node)
        if before_global_pool:
            insert_after_node = GraphStructureQuery.conv_insert_after_node_before_flatten(
                path_destination_to_source, flatten_node, gm,
            )
        else:
            insert_after_node = path_destination_to_source[flatten_index + 1]
        insert_index = path_destination_to_source.index(insert_after_node)
        consumer = path_destination_to_source[insert_index - 1] if insert_index > flatten_index else flatten_node

        new_out = _insert_call_module_after(gm, insert_after_node, name, insert_after_node)
        tail = _append_batchnorm_and_relu(gm, new_out, new_layer, name) if add_batchnorm_and_relu else new_out
        NodeEditor.swap_node_input(consumer, insert_after_node, tail)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def delete_layer(
        gm: fx.GraphModule,
        layer_id: str,
        input_shape: tuple[int, ...] | None = None,
    ) -> fx.GraphModule:
        """Remove *layer_id* and wire shape-compatible predecessor branches to successors only."""
        layer_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_module" and n.target == layer_id
        )
        input_layers = GraphStructureQuery.get_input_layers(layer_id, gm)
        output_layers = GraphStructureQuery.get_output_layers(layer_id, gm)
        output_shapes, input_shapes = LayerShapeAnalyser.collect_layer_shapes(
            gm, input_shape=input_shape
        )
        if is_merge_branch_layer(layer_node):
            remove_layer_from_sums(gm, layer_node)
        else:
            matching = compute_bypass_matching(input_layers, output_layers, output_shapes, input_shapes)
            if not output_layers:
                if not input_layers:
                    raise ValueError(f"Cannot delete {layer_id!r}: layer has no sequential neighbours")
                if not branch_only_bypass_compatible(layer_node, input_shapes):
                    raise ValueError(f"Cannot delete {layer_id!r}: no shape-compatible branch-only bypass")
                _rewire_branch_only_layer(layer_node)
            elif matching is None:
                raise ValueError(f"Cannot delete {layer_id!r}: no shape-compatible bypass matching")
            else:
                _rewire_layer_users(gm, layer_node, matching, output_layers)

        gm.graph.erase_node(layer_node)
        if hasattr(gm, layer_id):
            delattr(gm, layer_id)

        prune_unreachable_nodes(gm)
        gm.graph.lint()
        gm.recompile()
        return gm
