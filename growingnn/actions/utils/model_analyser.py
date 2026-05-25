import torch.nn as nn
import torch.fx as fx

from growingnn.core.config import EDITABLE_MODULES
from growingnn.core.logger import logger

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
    return True

def get_layer_module(target: fx.Node | str, gm: nn.Module | fx.GraphModule) -> nn.Module | None:
    """Resolve a submodule by FX node or by dotted module path (e.g. ``layer1.0.conv1``).

    Returns ``None`` if the path does not exist on ``gm``.
    Uses ``nn.Module.get_submodule`` so dotted paths are walked segment by segment,
    unlike ``getattr`` which treats the whole string as a single attribute name.
    """
    name = target.target if isinstance(target, fx.Node) else target
    try:
        return gm.get_submodule(str(name))
    except AttributeError:
        return None

def _is_editable_module(node: fx.Node, gm: fx.GraphModule) -> bool:
    if node.op != "call_module":
        logger.debug("node.target: %s is not an editable module", node.target)
        return False

    module = get_layer_module(node, gm)
    if module is None:
        logger.debug("node.target: %s could not be resolved on gm", node.target)
        return False

    for editable_module_type in EDITABLE_MODULES:
        if isinstance(module, editable_module_type):
            logger.debug(
                "node.target: %s is an editable module of type: %s",
                node.target,
                editable_module_type,
            )
            return True

    return False

def _is_at_least_one_hidden_module(n1: fx.Node, n2: fx.Node) -> bool:
    return _is_hidden_module(n1) or _is_hidden_module(n2)

def _is_edge_into_hidden_module(src: fx.Node, dst: fx.Node) -> bool:
    """True for visible→hidden and hidden→hidden edges; false otherwise."""
    src_hidden = _is_hidden_module(src)
    dst_hidden = _is_hidden_module(dst)
    return (not src_hidden and dst_hidden) or (src_hidden and dst_hidden)

def get_all_hidden_modules(model: nn.Module | fx.GraphModule) -> list[str]:
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    nodes: list[str] = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        if not _is_hidden_module(n):
            logger.debug("n.target: %s is not a hidden module", n.target)
            continue
        nodes.append(str(n.target))
    return nodes

def module_dependency_pairs(model: nn.Module | fx.GraphModule) -> list[tuple[str, str]]:
    """All ``(ancestor, descendant)`` pairs where the descendant is a hidden module reachable forward from the ancestor.

    For ``l1 -> l2 -> l3`` this yields ``(l1,l2)`` only (``l3`` is not hidden).
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
            if _is_editable_module(cur, gm) and _is_edge_into_hidden_module(n, cur):
                edges.append((src, str(cur.target)))
            stack.extend(cur.users)
    logger.debug("number of dependency pairs: %s", len(edges))
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
                edges.append((src, str(cur.target)))
                continue
            stack.extend(cur.users)
    logger.debug("number of sequential pairs: %s", len(edges))
    return list(dict.fromkeys(edges))

def get_amount_of_parameters(model: nn.Module | fx.GraphModule) -> int:
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    return sum(p.numel() for p in gm.parameters())



def _sequential_adj(model: nn.Module | fx.GraphModule) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    pred: dict[str, list[str]] = {}
    succ: dict[str, list[str]] = {}
    for a, b in dict.fromkeys(module_sequential_pairs(model)):
        pred.setdefault(b, []).append(a)
        succ.setdefault(a, []).append(b)
    return pred, succ


def get_input_layers(layer_id: str, model: nn.Module | fx.GraphModule) -> list[str]:
    pred, _ = _sequential_adj(model)
    return list(pred.get(layer_id, []))


def get_output_layers(layer_id: str, model: nn.Module | fx.GraphModule) -> list[str]:
    _, succ = _sequential_adj(model)
    return list(succ.get(layer_id, []))