import torch
import torch.nn as nn
import torch.fx as fx

from growingnn.actions.utils.layer_analyser import (
    is_passthrough, is_fork, is_add,
    node_output_width, inputs_match_width, all_sites_match_width,
)
from growingnn.actions.utils.model_analyser import get_layer_module
from growingnn.actions.utils.model_transformations import _find_call_module, replace_submodule
from growingnn.actions.utils.quaziIdentity import get_reshsper

_SIZABLE = (nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)


def _r(old, new, ref):
    return torch.tensor(get_reshsper(old, new), dtype=ref.dtype, device=ref.device)


def _vec(v, old, new):
    return (_r(old, new, v).T @ v).contiguous()


def reproject_linear_out(mod, n):
    """New Linear with out_features=n, weights re-projected from mod."""
    lin = nn.Linear(mod.in_features, n, bias=mod.bias is not None)
    with torch.no_grad():
        lin.weight.copy_((_r(mod.out_features, n, mod.weight).T @ mod.weight).contiguous())
        if mod.bias is not None:
            lin.bias.copy_(_vec(mod.bias, mod.out_features, n))
    return lin


def reproject_linear_in(mod, n):
    """New Linear with in_features=n, weights re-projected from mod."""
    lin = nn.Linear(n, mod.out_features, bias=mod.bias is not None)
    with torch.no_grad():
        lin.weight.copy_((mod.weight @ _r(mod.in_features, n, mod.weight)).contiguous())
        if mod.bias is not None:
            lin.bias.copy_(mod.bias)
    return lin


def reproject_bn(mod, n):
    """New BatchNorm with num_features=n, preserving BN1d vs BN2d type."""
    BN = type(mod)
    bn = BN(n, eps=mod.eps, momentum=mod.momentum,
            affine=mod.affine, track_running_stats=mod.track_running_stats)
    with torch.no_grad():
        for f in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(mod, f, None)
            if t is not None:
                getattr(bn, f).copy_(_vec(t, mod.num_features, n))
    return bn


def _reproject_out(mod, w):
    if isinstance(mod, nn.Linear): return reproject_linear_out(mod, w)
    return reproject_bn(mod, w)


def _shrink_out(gm, name, mod, w):
    if isinstance(mod, nn.Linear) and mod.out_features > w:
        replace_submodule(gm, name, reproject_linear_out(mod, w))
    elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)) and mod.num_features > w:
        replace_submodule(gm, name, reproject_bn(mod, w))


def _narrow_in(gm, name, mod, w):
    if isinstance(mod, nn.Linear):
        if mod.in_features != w and all_sites_match_width(gm, name, w):
            replace_submodule(gm, name, reproject_linear_in(mod, w))
    elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if mod.num_features != w:
            replace_submodule(gm, name, reproject_bn(mod, w))


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
        if mod is not None and isinstance(mod, _SIZABLE):
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
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if mod.num_features != w: replace_submodule(gm, name, reproject_bn(mod, w))
            _propagate(gm, user, w, seen)


# --------------- public API ---------------

def resize_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, new_width: int) -> fx.GraphModule:
    """Resize a layer's output to new_width and propagate the change through the graph."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = get_layer_module(layer_id, gm)
    if not isinstance(mod, _SIZABLE):
        raise TypeError(f"{layer_id} is {type(mod).__name__}, not a sizable module")
    replace_submodule(gm, layer_id, _reproject_out(mod, new_width))
    _propagate(gm, _find_call_module(gm.graph.nodes, layer_id), new_width, set())
    gm.recompile()
    return gm
