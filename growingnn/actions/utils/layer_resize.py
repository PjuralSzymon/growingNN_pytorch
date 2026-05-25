import torch.nn as nn
import torch.fx as fx

from growingnn.actions.utils.layer_analyser import (
    is_passthrough, is_fork, is_add,
    node_output_width, inputs_match_width, all_sites_match_width,
)
from growingnn.actions.utils.model_analyser import get_layer_module
from growingnn.actions.utils.model_transformations import _find_call_module, replace_submodule
from growingnn.actions.utils.layer_Factory import LinearFactory


def _shrink_out(gm, name, mod, w):
    if isinstance(mod, nn.Linear) and mod.out_features > w:
        replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_neurons(mod, w))


def _narrow_in(gm, name, mod, w):
    if isinstance(mod, nn.Linear) and mod.in_features != w and all_sites_match_width(gm, name, w):
        replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_connections(mod, w))


# --------------- graph traversal ---------------

def _sync(gm, node, w, seen, *, via_pass=False, at_add=None):
    key = ("s", node.name, w)
    if key in seen: return
    seen.add(key)
    if is_add(node):
        if not via_pass:
            for inp in node.all_input_nodes:
                _sync(gm, inp, w, seen, at_add=node)
        return
    if node.op == "call_module":
        mod = get_layer_module(node.target, gm)
        if isinstance(mod, nn.Linear):
            if is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            _shrink_out(gm, str(node.target), mod, w)
            _propagate(gm, node, w, seen)
            return
    if is_passthrough(gm, node):
        for pred in node.all_input_nodes:
            if is_fork(pred) and not is_passthrough(gm, pred): continue
            _sync(gm, pred, w, seen, via_pass=True)
        return
    if is_fork(node): return
    if node.op == "call_module":
        return


def _backward_in(gm, node, add_node, w, seen):
    if node in add_node.all_input_nodes or is_fork(node): return
    key = ("b", node.name, w)
    if key in seen: return
    seen.add(key)
    for pred in node.all_input_nodes:
        _backward_in(gm, pred, add_node, w, seen)
    if node.op == "call_module" and inputs_match_width(gm, node, w):
        _narrow_in(gm, str(node.target), get_layer_module(node.target, gm), w)


def _propagate(gm, node, w, seen):
    key = ("p", node.name, w)
    if key in seen: return
    seen.add(key)
    if is_fork(node) and node_output_width(gm, node) != w: return
    for user in list(node.users):
        if user.op == "output": continue
        if is_add(user):
            for inp in user.all_input_nodes:
                if inp is not node:
                    _sync(gm, inp, w, seen, at_add=user)
            if not is_fork(node):
                for pred in node.all_input_nodes:
                    _backward_in(gm, pred, user, w, seen)
            _propagate(gm, user, w, seen)
            continue
        if is_passthrough(gm, user):
            _propagate(gm, user, w, seen)
            continue
        if user.op != "call_module":
            continue
        mod = get_layer_module(user.target, gm)
        if mod is None:
            continue
        name = str(user.target)
        if isinstance(mod, nn.Linear):
            if not inputs_match_width(gm, user, w): continue
            _narrow_in(gm, name, mod, w)
            _propagate(gm, user, get_layer_module(name, gm).out_features, seen)


# --------------- public API ---------------

def resize_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, new_width: int) -> fx.GraphModule:
    """Resize a Linear layer's output to new_width and propagate the change through the graph."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is {type(mod).__name__}, not nn.Linear")
    replace_submodule(gm, layer_id, LinearFactory.create_linear_with_rescaled_neurons(mod, new_width))
    _propagate(gm, _find_call_module(gm.graph.nodes, layer_id), new_width, set())
    gm.recompile()
    return gm
