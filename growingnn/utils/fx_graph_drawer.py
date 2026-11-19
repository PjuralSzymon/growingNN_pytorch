import torch.nn as nn
from torch.fx.passes.graph_drawer import FxGraphDrawer
from graphviz import Digraph


def module_weight_shape_suffix(mod: nn.Module) -> str:
    """Extra graphviz label fragment for modules whose `.weight` shape is informative."""
    if isinstance(mod, nn.Linear):
        return f"\\nweight {tuple(mod.weight.shape)}"
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return f"\\nweight {tuple(mod.weight.shape)}"
    return ""


def draw_torch_fx_graph(gm, output_file="fx_graph", fmt="svg"):
    fmt = fmt.lower().removeprefix(".")
    dot = FxGraphDrawer(gm, "model").get_dot_graph()
    path = f"{output_file}.{fmt}"
    writer = getattr(dot, f"write_{fmt}", None)
    if writer is None:
        raise ValueError(
            f"Unsupported graph format {fmt!r}. "
            f"Use a pydot-supported name (e.g. svg, png, pdf, jpg)."
        )
    writer(path)

def draw_filtered_fx_graph(
    gm,
    output_file="fx_graph",
    fmt="svg",
    *,
    include_functions=True,
    module_targets=None,
    compact_labels=False,
    rankdir="LR",
):
    """Draw a readable FX dependency graph with optional functional-operation nodes.

    When ``include_functions`` is false, only module calls are drawn and paths
    through functional/method nodes are collapsed to direct module dependencies.
    The default preserves the historical operation-level view.
    """
    kept_ops = {"call_module"}
    if include_functions:
        kept_ops.add("call_function")
    dot = Digraph(name="FXGraph")
    dot.attr(rankdir=rankdir)
    dot.attr("node", shape="box", fontsize="10")

    def is_kept(node):
        if node.op not in kept_ops:
            return False
        if node.op == "call_module" and module_targets is not None:
            return str(node.target) in module_targets
        return True

    kept_nodes = [n for n in gm.graph.nodes if is_kept(n)]
    kept_names = {n.name for n in kept_nodes}

    def find_kept_parents(node, visited=None):
        """
        Walk backward through input nodes until we reach kept nodes.
        This preserves connectivity even when placeholder/call_method/output
        nodes are omitted.
        """
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)

        if is_kept(node):
            return {node}

        parents = set()
        for inp in node.all_input_nodes:
            # This graph is a DAG: a node visited through one input path has
            # already contributed all of its nearest kept parents. Reusing the
            # same set avoids exponential revisits when many functional paths
            # share ancestors and the filtered view hides intermediate nodes.
            parents |= find_kept_parents(inp, visited)
        return parents

    def short_label(node):
        if node.op == "call_module":
            target = str(node.target)
            if compact_labels:
                parts = target.split(".")
                block_index = next((i for i, part in enumerate(parts[:-1])
                                    if part == "h" and parts[i + 1].isdigit()), None)
                if block_index is not None:
                    target = ".".join(parts[block_index:])
                elif "transformer" in parts:
                    target = ".".join(parts[parts.index("transformer") + 1:])
                else:
                    target = ".".join(parts[-3:])
            base = f"module\\n{target}"
            mod = gm.get_submodule(str(node.target))
            return base + module_weight_shape_suffix(mod)
        if node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            return f"function\\n{name}"
        return f"{node.op}\\n{node.target}"

    # add only kept nodes
    for node in kept_nodes:
        dot.node(node.name, short_label(node))

    # connect each kept node to the nearest previous kept nodes
    for node in kept_nodes:
        direct_inputs = list(node.all_input_nodes)
        parent_kept = set()

        for inp in direct_inputs:
            parent_kept |= find_kept_parents(inp)

        for parent in parent_kept:
            if parent.name in kept_names and parent.name != node.name:
                dot.edge(parent.name, node.name)

    dot.render(output_file, format=fmt, cleanup=True)
    return dot
