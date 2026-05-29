"""Graph-level edits: add/remove layers and rewire connections."""

from __future__ import annotations

import operator

import torch.fx as fx

from growingnn.utils.fx.node_analysis import ModuleResolver
from growingnn.utils.fx.node_editor import NodeEditor


def _insert_call_module_after(gm, insert_after, module_name, module_input):
    """Insert a new call_module node immediately after the given node in the graph."""
    with gm.graph.inserting_after(insert_after):
        return gm.graph.call_module(module_name, args=(module_input,))


def _path_dst_to_src(dst, src, seen=None):
    """Return a list of nodes forming a path from dst back to src, or None if unreachable."""
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
        nodes = list(gm.graph.nodes)
        gm.add_module(name, new_layer)

        src = ModuleResolver.find_call_module(nodes, src_name)
        dst = ModuleResolver.find_call_module(nodes, dst_name)

        new_out = _insert_call_module_after(gm, dst, name, src)

        with gm.graph.inserting_after(new_out):
            added = gm.graph.call_function(operator.add, args=(dst, new_out))

        dst.replace_all_uses_with(added)
        added.args = (dst, new_out)

        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def add_new_seq_layer(gm, src_name, dst_name, new_layer, name):
        """Insert *new_layer* sequentially on the path from *src_name* to *dst_name*."""
        nodes = list(gm.graph.nodes)
        gm.add_module(name, new_layer)

        src = ModuleResolver.find_call_module(nodes, src_name)
        dst = ModuleResolver.find_call_module(nodes, dst_name)
        if src is dst:
            raise ValueError("src and dst must differ.")

        path = _path_dst_to_src(dst, src)
        if path is None:
            raise ValueError(f"No path from {dst_name!r} back to {src_name!r} in the FX graph.")

        src = path[1]
        new_out = _insert_call_module_after(gm, src, name, src)
        NodeEditor.swap_node_input(dst, src, new_out)

        gm.graph.lint()
        gm.recompile()

    @staticmethod
    def delete_layer(gm: fx.GraphModule, layer_id: str) -> fx.GraphModule:
        """Remove *layer_id* from the graph, wiring its inputs directly to its users."""
        graph = gm.graph

        layer_node = next(
            n for n in graph.nodes
            if n.op == "call_module" and n.target == layer_id
        )

        input_nodes = list(layer_node.all_input_nodes)
        output_nodes = list(layer_node.users)

        new_input = input_nodes[0]
        for input_node in input_nodes[1:]:
            if input_node is new_input:
                continue
            with gm.graph.inserting_after(new_input):
                new_input = gm.graph.call_function(
                    operator.add,
                    args=(new_input, input_node),
                )
        for output_node in output_nodes:
            output_node.replace_input_with(layer_node, new_input)

        graph.erase_node(layer_node)
        if hasattr(gm, layer_id):
            delattr(gm, layer_id)

        graph.lint()
        gm.recompile()
        return gm
