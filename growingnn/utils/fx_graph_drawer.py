from torch.fx.passes.graph_drawer import FxGraphDrawer
from graphviz import Digraph

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

def draw_filtered_fx_graph(gm, output_file="fx_graph", fmt="svg"):
    kept_ops = {"call_module", "call_function"}
    dot = Digraph(name="FXGraph")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", fontsize="10")

    kept_nodes = [n for n in gm.graph.nodes if n.op in kept_ops]
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

        if node.op in kept_ops:
            return {node}

        parents = set()
        for inp in node.all_input_nodes:
            parents |= find_kept_parents(inp, visited.copy())
        return parents

    def short_label(node):
        if node.op == "call_module":
            return f"module\\n{node.target}"
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