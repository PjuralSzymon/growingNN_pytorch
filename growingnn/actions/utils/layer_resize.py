import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from growingnn.actions.utils.model_analyser import get_layer_module
from growingnn.actions.utils.model_transformations import _find_call_module
from growingnn.actions.utils.quaziIdentity import get_reshsper

_PASS = (nn.Dropout, nn.Identity, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.ELU, nn.Sigmoid)
_ACT_FN = {F.relu, F.gelu, F.silu, F.tanh, F.elu, F.sigmoid, torch.relu, torch.sigmoid, torch.tanh}


def _r(old, new, ref):
    return torch.tensor(get_reshsper(old, new), dtype=ref.dtype, device=ref.device)


def _vec(v, old, new):
    return (_r(old, new, v).T @ v).contiguous()


def _set(gm, name, mod):
    parent, _, leaf = name.rpartition(".")
    (getattr(gm, parent) if parent else gm).add_module(leaf, mod)


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
    """New BatchNorm1d with num_features=n, stats re-projected from mod."""
    bn = nn.BatchNorm1d(n, eps=mod.eps, momentum=mod.momentum,
                        affine=mod.affine, track_running_stats=mod.track_running_stats)
    with torch.no_grad():
        for f in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(mod, f, None)
            if t is not None:
                getattr(bn, f).copy_(_vec(t, mod.num_features, n))
    return bn


def _is_pass(gm, n):
    return (n.op == "call_function" and n.target in _ACT_FN) or \
           (n.op == "call_module" and isinstance(get_layer_module(n.target, gm), _PASS))


def _is_fork(n): return len(n.users) > 1
def _is_add(n): return n.op == "call_function" and n.target == operator.add


def _width(gm, n):
    """Output channel width of a node, or None if unknown."""
    if n.op == "call_module":
        m = get_layer_module(n.target, gm)
        if isinstance(m, nn.Linear): return m.out_features
        if isinstance(m, nn.BatchNorm1d): return m.num_features
    if (_is_pass(gm, n) or _is_add(n)) and n.all_input_nodes:
        return _width(gm, n.all_input_nodes[0])
    return None


def _inputs_at(gm, n, w):
    return n.all_input_nodes and all(_width(gm, i) == w for i in n.all_input_nodes)


def _all_sites_at(gm, name, w):
    return all(_inputs_at(gm, n, w)
               for n in gm.graph.nodes if n.op == "call_module" and n.target == name)


def _shrink_out(gm, name, mod, w):
    if isinstance(mod, nn.Linear) and mod.out_features > w:
        _set(gm, name, reproject_linear_out(mod, w))
    elif isinstance(mod, nn.BatchNorm1d) and mod.num_features > w:
        _set(gm, name, reproject_bn(mod, w))


def _narrow_in(gm, name, mod, w):
    if isinstance(mod, nn.Linear) and mod.in_features != w and _all_sites_at(gm, name, w):
        _set(gm, name, reproject_linear_in(mod, w))
    elif isinstance(mod, nn.BatchNorm1d) and mod.num_features != w:
        _set(gm, name, reproject_bn(mod, w))


def _sync(gm, node, w, seen, *, via_pass=False, at_add=None):
    key = ("s", node.name, w)
    if key in seen: return
    seen.add(key)
    if _is_add(node):
        if not via_pass:
            for inp in node.all_input_nodes:
                _sync(gm, inp, w, seen, at_add=node)
        return
    if node.op == "call_module":
        mod = get_layer_module(node.target, gm)
        if isinstance(mod, (nn.Linear, nn.BatchNorm1d)):
            if _is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            _shrink_out(gm, str(node.target), mod, w)
            _propagate(gm, node, w, seen)
            return
    if _is_pass(gm, node):
        for pred in node.all_input_nodes:
            if _is_fork(pred) and not _is_pass(gm, pred): continue
            _sync(gm, pred, w, seen, via_pass=True)
        return
    if _is_fork(node): return
    raise NotImplementedError(f"unsupported branch node {node.op} {node.target}")


def _backward_in(gm, node, add_node, w, seen):
    if node in add_node.all_input_nodes or _is_fork(node): return
    key = ("b", node.name, w)
    if key in seen: return
    seen.add(key)
    for pred in node.all_input_nodes:
        _backward_in(gm, pred, add_node, w, seen)
    if node.op == "call_module" and _inputs_at(gm, node, w):
        _narrow_in(gm, str(node.target), get_layer_module(node.target, gm), w)


def _propagate(gm, node, w, seen):
    key = ("p", node.name, w)
    if key in seen: return
    seen.add(key)
    if _is_fork(node) and _width(gm, node) != w: return
    for user in list(node.users):
        if user.op == "output": continue
        if _is_add(user):
            for inp in user.all_input_nodes:
                if inp is not node:
                    _sync(gm, inp, w, seen, at_add=user)
            if not _is_fork(node):
                for pred in node.all_input_nodes:
                    _backward_in(gm, pred, user, w, seen)
            _propagate(gm, user, w, seen)
            continue
        if _is_pass(gm, user):
            _propagate(gm, user, w, seen)
            continue
        if user.op != "call_module":
            raise NotImplementedError(f"unsupported user {user.op} {user.target}")
        mod = get_layer_module(user.target, gm)
        name = str(user.target)
        if isinstance(mod, nn.Linear):
            if not _inputs_at(gm, user, w): continue
            _narrow_in(gm, name, mod, w)
            _propagate(gm, user, get_layer_module(name, gm).out_features, seen)
        elif isinstance(mod, nn.BatchNorm1d):
            if mod.num_features != w: _set(gm, name, reproject_bn(mod, w))
            _propagate(gm, user, w, seen)
        else:
            raise NotImplementedError(f"unsupported module {type(mod)}")


def resize_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, new_width: int) -> fx.GraphModule:
    """Resize a layer's output to new_width and propagate the change through the graph."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = get_layer_module(layer_id, gm)
    if not isinstance(mod, (nn.Linear, nn.BatchNorm1d)):
        raise TypeError(f"{layer_id} is not nn.Linear or nn.BatchNorm1d")
    if isinstance(mod, nn.Linear):
        _set(gm, layer_id, reproject_linear_out(mod, new_width))
    else:
        _set(gm, layer_id, reproject_bn(mod, new_width))
    _propagate(gm, _find_call_module(gm.graph.nodes, layer_id), new_width, set())
    gm.recompile()
    return gm
