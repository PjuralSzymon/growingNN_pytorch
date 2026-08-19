"""Graph-level edits: add/remove layers and rewire connections."""

from __future__ import annotations

import torch.fx as fx

from growingnn.core.config import (
    TRANSFORMER_PACKED_PROJECTION_WIDTH_SENSITIVE_FUNCTIONS,
    TRANSFORMER_PACKED_PROJECTION_WIDTH_SENSITIVE_METHODS,
)
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


def user_requires_exact_output_shape(user: fx.Node) -> bool:
    """True for Transformer packed-projection users (split/view/slice), not relu/add."""
    if user.op == "call_method":
        return user.target in TRANSFORMER_PACKED_PROJECTION_WIDTH_SENSITIVE_METHODS
    if user.op == "call_function":
        name = getattr(user.target, "__name__", "")
        return name in TRANSFORMER_PACKED_PROJECTION_WIDTH_SENSITIVE_FUNCTIONS
    return False


def bypass_valid_for_all_users(
    layer_node: fx.Node,
    replacement_shape: tuple[int, ...] | None,
    layer_output_shape: tuple[int, ...] | None,
) -> bool:
    """True when replacement_shape can replace layer_node's output for every immediate user."""
    for user in layer_node.users:
        if not user_requires_exact_output_shape(user):
            continue
        if replacement_shape is None or layer_output_shape is None:
            return False
        if replacement_shape != layer_output_shape:
            return False
    return True


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


def _rewire_users_of_deleted_non_merge_layer_or_raise_if_bypass_is_not_shape_safe(
    gm: fx.GraphModule,
    layer_node: fx.Node,
    layer_id: str,
    input_layers: list[str],
    output_layers: list[str],
    output_shapes: dict[str, tuple[int, ...]],
    input_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Rewire users of a non-merge layer, or raise if no shape-safe bypass exists."""
    layer_out = output_shapes.get(layer_id)
    if not all(
        bypass_valid_for_all_users(layer_node, output_shapes.get(pred_id), layer_out)
        for pred_id in input_layers
    ):
        raise ValueError(
            f"Cannot delete {layer_id!r}: replacement not valid for all users of this layer"
        )
    matching = compute_bypass_matching(input_layers, output_layers, output_shapes, input_shapes)
    if not output_layers:
        if not input_layers:
            raise ValueError(f"Cannot delete {layer_id!r}: layer has no sequential neighbours")
        if not branch_only_bypass_compatible(layer_node, input_shapes):
            raise ValueError(f"Cannot delete {layer_id!r}: no shape-compatible branch-only bypass")
        _rewire_branch_only_layer(layer_node)
        return
    if matching is None:
        raise ValueError(f"Cannot delete {layer_id!r}: no shape-compatible bypass matching")
    _rewire_layer_users(gm, layer_node, matching, output_layers)


class ModelStructureEditor:
    """Add and remove layers in an FX graph."""

    @staticmethod
    def add_new_residual_layer(gm, src_name, dst_name, new_layer, name):
        """Insert *new_layer* as a residual branch from *src_name* added to *dst_name* output."""
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
    def add_new_seq_layer(gm, src_name, dst_name, new_layer, name):
        """Insert *new_layer* sequentially on the path from *src_name* to *dst_name*."""
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
        NodeEditor.swap_node_input(dst, path[1], new_out)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def add_new_seq_layer_before_flatten(gm, src_name, dst_name, new_layer, name):
        """Insert *new_layer* immediately before the flatten node on the src→dst sequential path."""
        from growingnn.utils.fx.graph_analysis import GraphStructureQuery

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

        flatten_index = path_destination_to_source.index(flatten_node)
        if flatten_index + 1 >= len(path_destination_to_source):
            raise ValueError("Flatten has no predecessor on the sequential path.")
        insert_after_node = path_destination_to_source[flatten_index + 1]
        pools_before_flatten = GraphStructureQuery.find_pool_nodes_between_flatten_and_source(
            path_destination_to_source, flatten_node, gm,
        )
        if not pools_before_flatten:
            raise ValueError("Sequential conv before flatten requires at least one pool before flatten.")

        new_out = _insert_call_module_after(gm, insert_after_node, name, insert_after_node)
        NodeEditor.swap_node_input(flatten_node, insert_after_node, new_out)
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
            _rewire_users_of_deleted_non_merge_layer_or_raise_if_bypass_is_not_shape_safe(
                gm, layer_node, layer_id, input_layers, output_layers, output_shapes, input_shapes,
            )

        gm.graph.erase_node(layer_node)
        if hasattr(gm, layer_id):
            delattr(gm, layer_id)

        prune_unreachable_nodes(gm)
        gm.graph.lint()
        gm.recompile()
        return gm
