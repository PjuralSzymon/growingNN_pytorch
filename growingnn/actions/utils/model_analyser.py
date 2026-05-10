import torch.nn as nn
import torch.fx as fx

from growingnn.config import EDITABLE_MODULES

def is_internal_call_module(node: fx.Node) -> bool:
    return (
        node.op == "call_module"
        and not any(inp.op == "placeholder" for inp in node.all_input_nodes)
        and not any(user.op == "output" for user in node.users)
    )

def _is_hidden_module(node: fx.Node) -> bool:
    if len(node.users)==0 or len(node.all_input_nodes) == 0:
        return False
    if any(user.op == "output" for user in node.users):
        return False

    if "placeholder" in node.all_input_nodes:
        return False

    if len(node.all_input_nodes) == 0:
        return False
    if len(node.all_input_nodes) == 1 and node.all_input_nodes[0] == None:
        return False

    if len(node.all_input_nodes) == 1:
        if node.all_input_nodes[0] == None:
            return False
        if len(node.all_input_nodes[0].all_input_nodes) == 0:
            return False

    has_real_input = any(
            not inp.op == "placeholder"
            for inp in node.all_input_nodes
        )

    has_real_user = any(
        not user.op == "output"
        for user in node.users
    )
    return True

def _is_editable_module(node: fx.Node, gm: fx.GraphModule) -> bool:
    if node.op != "call_module":
        print(f"node.target: {node.target} is not a editable module")
        return False
    
    layer_module = getattr(gm, str(node.target), None)
    if layer_module is None:
        print(f"layer_module: {layer_module} is not a editable module")
        return False

    module = gm.get_submodule(str(node.target))

    for editable_module_type in EDITABLE_MODULES:
        if isinstance(module, editable_module_type):
            print(f"node.target: {node.target} is a editable module of type: {editable_module_type}")
            return True
        else:
            print(f"node.target: {node.target} is not a editable module is type: {type(module)}")
    return False

def _is_at_least_one_hidden_module(n1: fx.Node, n2: fx.Node) -> bool:
    return _is_hidden_module(n1) or _is_hidden_module(n2)

def get_all_hidden_modules(model: nn.Module | fx.GraphModule) -> list[str]:
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    nodes: list[str] = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        if not _is_hidden_module(n):
            print(f"n.target: {n.target} is not a hidden module")
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
        if not _is_editable_module(n, gm):
            continue
        src = str(n.target)
        stack, seen = list(n.users), set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            print(f"cur: {cur.target} is hidden: {_is_hidden_module(cur)}")
            if _is_editable_module(cur, gm) and _is_hidden_module(cur):
                print(f" adding pair: {src} -> {cur.target}")
                edges.append((src, str(cur.target)))
            stack.extend(cur.users)
    print(f"edges: {edges}")
    print(f"edges 2: {list(dict.fromkeys(edges))}")
    return list(dict.fromkeys(edges))
    

def module_sequential_pairs(model: nn.Module | fx.GraphModule) -> list[tuple[str, str]]:
    """
    This function returns all ``(ancestor, descendant)`` pairs that are next to each other in the model.
    For ``l1 -> l2 -> l3`` this yields ``(l1,l2), (l2,l3)``.
    """
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    edges: list[tuple[str, str]] = []
    for n in gm.graph.nodes:
        if not _is_editable_module(n, gm):
            continue
        src = str(n.target)
        stack, seen = list(n.users), set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if _is_editable_module(cur, gm) and _is_at_least_one_hidden_module(n, cur):
                print(f" adding pair: {src} -> {cur.target}")
                edges.append((src, str(cur.target)))
                continue
            stack.extend(cur.users)
    print(f"edges: {edges}")
    print(f"edges 2: {list(dict.fromkeys(edges))}")
    return list(dict.fromkeys(edges))

def get_amount_of_parameters(model: nn.Module | fx.GraphModule) -> int:
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    return sum(p.numel() for p in gm.parameters())