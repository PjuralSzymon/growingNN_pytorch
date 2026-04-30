import torch.nn as nn
import torch.fx as fx


def _has_module(
    start: fx.Node,
    next_nodes: callable,
    seen: set[fx.Node] | None = None,
) -> bool:
    """
    Generic graph traversal.

    Starting from `start`, walk through the graph using `next_nodes`
    (which defines the direction: upstream or downstream).

    Return True if we encounter any `call_module` node along the way.
    """
    # Track visited nodes to avoid infinite loops in cyclic graphs
    seen = seen or set()
    if start in seen:
        return False
    seen.add(start)

    # Explore neighboring nodes (inputs or users depending on direction)
    for n in next_nodes(start):
        # If we directly hit a module → success
        # Otherwise, keep searching recursively
        if n.op == "call_module" or _has_module(n, next_nodes, seen):
            return True

    # No module found in this direction
    return False


def _has_module_upstream(node: fx.Node) -> bool:
    """
    Check if there is any `call_module` BEFORE this node in the graph.
    Uses `.all_input_nodes` → walks backward (toward inputs).
    """
    return _has_module(node, lambda n: n.all_input_nodes)


def _has_module_downstream(node: fx.Node) -> bool:
    """
    Check if there is any `call_module` AFTER this node in the graph.
    Uses `.users` → walks forward (toward outputs).
    """
    return _has_module(node, lambda n: n.users)

def _is_hidden_module(node: fx.Node) -> bool:
    return _has_module_upstream(node) and _has_module_downstream(node)

def get_all_hidden_modules(model: nn.Module | fx.GraphModule) -> list[str]:
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    nodes: list[str] = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        if not _is_hidden_module(n):
            continue
        nodes.append(str(n.target))
    return nodes

def module_dependency_pairs(model: nn.Module | fx.GraphModule) -> list[tuple[str, str]]:
    """All ``(ancestor, descendant)`` pairs where the descendant module is reachable forward from the ancestor.

    For ``l1 -> l2 -> l3`` this yields ``(l1,l2), (l1,l3), (l2,l3)``.
    """
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    edges: list[tuple[str, str]] = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        src = str(n.target)
        stack, seen = list(n.users), set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur.op == "call_module":
                edges.append((src, str(cur.target)))
            stack.extend(cur.users)
    return list(dict.fromkeys(edges))
    

def module_sequential_pairs(model: nn.Module | fx.GraphModule) -> list[tuple[str, str]]:
    """
    This function returns all ``(ancestor, descendant)`` pairs that are next to each other in the model.
    For ``l1 -> l2 -> l3`` this yields ``(l1,l2), (l2,l3)``.
    """
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    edges: list[tuple[str, str]] = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        src = str(n.target)
        stack, seen = list(n.users), set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur.op == "call_module":
                edges.append((src, str(cur.target)))
                continue
            stack.extend(cur.users)
    return list(dict.fromkeys(edges))
