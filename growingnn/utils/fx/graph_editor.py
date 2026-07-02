"""Graph-level edits: add/remove layers and rewire connections."""

from __future__ import annotations

import torch.fx as fx

from growingnn.utils.fx.node_analysis import ModuleResolver
from growingnn.utils.fx.node_editor import NodeEditor
from growingnn.utils.fx.sum_nodes import connect_residual_branch, sum_nodes


def _insert_call_module_after(gm, insert_after, module_name, module_input):
    with gm.graph.inserting_after(insert_after):
        return gm.graph.call_module(module_name, args=(module_input,))


def _path_dst_to_src(dst, src, seen=None):
    if dst is src:
        return [src]
    if seen is None:
        seen = set()
    if dst in seen:
        return None
    seen.add(dst)
    for pred in dst.all_input_nodes:
        tail = _path_dst_to_src(pred, src, seen)
        if tail is not None:
            return [dst] + tail
    seen.discard(dst)
    return None


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

        path = _path_dst_to_src(dst, src)
        if path is None:
            raise ValueError(f"No path from {dst_name!r} back to {src_name!r} in the FX graph.")

        new_out = _insert_call_module_after(gm, path[1], name, path[1])
        NodeEditor.swap_node_input(dst, path[1], new_out)
        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def delete_layer(gm: fx.GraphModule, layer_id: str) -> fx.GraphModule:
        """Remove *layer_id* from the graph, wiring its inputs directly to its users."""
        layer_node = next(
            n for n in gm.graph.nodes
            if n.op == "call_module" and n.target == layer_id
        )
        inputs = list(layer_node.all_input_nodes)
        replacement = inputs[0] if len(inputs) == 1 else sum_nodes(gm, inputs)
        for user in list(layer_node.users):
            user.replace_input_with(layer_node, replacement)
        if len(inputs) > 1:
            replacement.args = tuple(inputs)

        gm.graph.erase_node(layer_node)
        if hasattr(gm, layer_id):
            delattr(gm, layer_id)

        gm.graph.lint()
        gm.recompile()
        return gm
