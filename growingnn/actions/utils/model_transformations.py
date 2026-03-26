import operator
import torch

def _find_call_module(nodes, target_name):
    for n in nodes:
        if n.op == "call_module" and n.target == target_name:
            return n
    available = sorted({n.target for n in nodes if n.op == "call_module"})
    raise ValueError(
        f"No call_module node with target {target_name!r}. "
        f"Available call_module targets: {available}"
    )


def add_new_residual_layer(gm, src_name, dst_name, new_layer, name="res_layer"):
    nodes = list(gm.graph.nodes)

    gm.add_module(name, new_layer)

    src = _find_call_module(nodes, src_name)
    dst = _find_call_module(nodes, dst_name)

    with gm.graph.inserting_after(dst):
        new_out = gm.graph.call_module(name, args=(src,))

    with gm.graph.inserting_after(new_out):
        added = gm.graph.call_function(operator.add, args=(dst, new_out))

    dst.replace_all_uses_with(added)
    added.args = (dst, new_out)

    gm.graph.lint()
    gm.recompile()