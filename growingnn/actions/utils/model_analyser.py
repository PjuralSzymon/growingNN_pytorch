import torch.nn as nn
import torch.fx as fx


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
