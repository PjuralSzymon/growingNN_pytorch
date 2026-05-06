import operator

from torch import fx

# Common functions: 

def _insert_call_module_after(gm, insert_after, module_name, module_input):
    with gm.graph.inserting_after(insert_after):
        return gm.graph.call_module(module_name, args=(module_input,))

# Residual layers: 

def _find_call_module(nodes, target_name):
    for n in nodes:
        if n.op == "call_module" and n.target == target_name:
            return n
    available = sorted({n.target for n in nodes if n.op == "call_module"})
    raise ValueError(
        f"No call_module node with target {target_name!r}. "
        f"Available call_module targets: {available}"
    )


def add_new_residual_layer(gm, src_name, dst_name, new_layer, name):
    nodes = list(gm.graph.nodes)

    gm.add_module(name, new_layer)

    src = _find_call_module(nodes, src_name)
    dst = _find_call_module(nodes, dst_name)

    new_out = _insert_call_module_after(gm, dst, name, src)

    with gm.graph.inserting_after(new_out):
        added = gm.graph.call_function(operator.add, args=(dst, new_out))

    dst.replace_all_uses_with(added)
    added.args = (dst, new_out)

    gm.graph.lint()
    gm.recompile()

# Sequential layers: 

def _path_dst_to_src(dst, src, seen=None):
    """Backtrack from ``dst`` along inputs until ``src``; return ``[dst, …, src]`` or ``None``."""
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


def _replace_node_input(node, old, new):
    if old in node.args:
        node.args = tuple(new if a is old else a for a in node.args)
    if node.kwargs and old in node.kwargs.values():
        node.kwargs = {k: (new if v is old else v) for k, v in node.kwargs.items()}


def add_new_seq_layer(gm, src_name, dst_name, new_layer, name):
    """Insert ``new_layer`` on the input path from ``dst`` back to ``src`` (e.g. ``l1→ReLU→l2``)."""
    nodes = list(gm.graph.nodes)
    gm.add_module(name, new_layer)

    src = _find_call_module(nodes, src_name)
    dst = _find_call_module(nodes, dst_name)
    if src is dst:
        raise ValueError("src and dst must differ.")

    path = _path_dst_to_src(dst, src)
    if path is None:
        raise ValueError(f"No path from {dst_name!r} back to {src_name!r} in the FX graph.")

    src = path[1]

    new_out = _insert_call_module_after(gm, src, name, src)

    _replace_node_input(dst, src, new_out)

    gm.graph.lint()
    gm.recompile()


def delete_layer(gm: fx.GraphModule, layer_id: str) -> fx.GraphModule:
    graph = gm.graph

    layer_node = next(
        n for n in graph.nodes
        if n.op == "call_module" and n.target == layer_id
    )

    input_nodes = list(layer_node.all_input_nodes)
    output_nodes = list(layer_node.users)

    # Connect every input of deleted layer to every output of deleted layer
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

    # Remove deleted layer node from graph
    graph.erase_node(layer_node)
    if hasattr(gm, layer_id):
        delattr(gm, layer_id)

    graph.lint()
    gm.recompile()

    return gm