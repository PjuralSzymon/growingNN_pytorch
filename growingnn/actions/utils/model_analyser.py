import torch.nn as nn
import torch.fx as fx


def _has_module(nodes) -> bool:
    if isinstance(nodes, list):
        print(f"List node ! nodes: {nodes} - {len(nodes)} type: {type(nodes)}")
        if len(nodes) == 0:
            return False
        if len(nodes) == 1 and nodes[0] == None:
            return False
        return True
    elif isinstance(nodes, dict):
        print(f"Dict node ! nodes: {nodes} - {len(nodes)} type: {type(nodes)}")
        if len(nodes) == 0:
            return False
        if len(nodes) == 1 and nodes[list(nodes.keys())[0]] == None:
            return False
        return True

def _has_module_upstream(node: fx.Node) -> bool:
    return _has_module(node.all_input_nodes)

def _has_module_downstream(node: fx.Node) -> bool:
    p
    return _has_module(node.users)


def is_internal_call_module(node: fx.Node) -> bool:
    return (
        node.op == "call_module"
        and not any(inp.op == "placeholder" for inp in node.all_input_nodes)
        and not any(user.op == "output" for user in node.users)
    )

def _is_hidden_module(node: fx.Node) -> bool:
    if len(node.users)==0 or len(node.all_input_nodes) == 0:
        print(f"node: {node.target} has no users or all_input_nodes")
        return False
    if any(user.op == "output" for user in node.users):
        print(f"node: {node.target} has output is users")
        return False

    if "placeholder" in node.all_input_nodes:
        print(f"node: {node.target} has placeholder is all_input_nodes")
        return False

    if len(node.all_input_nodes) == 0:
        print(f"node: {node.target} has no all_input_nodes")
        return False
    if len(node.all_input_nodes) == 1 and node.all_input_nodes[0] == None:
        print(f"node: {node.target} has no all_input_nodes")
        return False

    if len(node.all_input_nodes) == 1:
        if node.all_input_nodes[0] == None:
            print(f"node: {node.target} all_input_nodes has only 1 value and it is None")
            return False
        if len(node.all_input_nodes[0].all_input_nodes) == 0:
            print(f"node: {node.target} all_input_nodes has only 1 value and it has no all_input_nodes")
            return False

    has_real_input = any(
            not inp.op == "placeholder"
            for inp in node.all_input_nodes
        )

    has_real_user = any(
        not user.op == "output"
        for user in node.users
    )
    print(f"{node.target} -------------------------")
    print(f"node.users: {node.users}")
    print(f"user keyrs: {str(list(node.users.keys()))}")
    print(f"node.op: {node.op}")
    print(f"has_real_input: {has_real_input}")
    print(f"has_real_user: {has_real_user}")
    print(f"node.all_input_nodes: {node.all_input_nodes}")
    for subNode in node.all_input_nodes:
        print(f"subNode: {subNode.target}")
        print(f"subNode.users: {subNode.users}")
        print(f"subNode.all_input_nodes: {subNode.all_input_nodes}")
    print("-------------------------")
    print(f"node: {node.target} is a hidden module")
    return True

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
        if n.op != "call_module":
            continue
        src = str(n.target)
        stack, seen = list(n.users), set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            print(f"cur: {cur.target} is hidden: {_is_hidden_module(cur)}")
            if cur.op == "call_module" and _is_hidden_module(cur):
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
            if cur.op == "call_module" and _is_at_least_one_hidden_module(n, cur):
                edges.append((src, str(cur.target)))
                continue
            stack.extend(cur.users)
    return list(dict.fromkeys(edges))
