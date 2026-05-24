import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from growingnn.actions.utils.model_analyser import get_layer_module
from growingnn.actions.utils.quaziIdentity import get_reshsper

_PASS = (nn.Dropout, nn.Identity, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.ELU, nn.Sigmoid)
_ACT_FN = {F.relu, F.gelu, F.silu, F.tanh, F.elu, F.sigmoid, torch.relu, torch.sigmoid, torch.tanh}


def _gm(model):
    return model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)


def _node(gm, layer_id):
    return next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == layer_id)


def _r(old, new, ref):
    return torch.tensor(get_reshsper(old, new), dtype=ref.dtype, device=ref.device)


def _set_module(gm, name, mod):
    parent, _, leaf = name.rpartition(".")
    (getattr(gm, parent) if parent else gm).add_module(leaf, mod)


def _vec(v, old, new):
    return (_r(old, new, v).T @ v).contiguous()


def _linear_out(old: nn.Linear, new_out: int) -> nn.Linear:
    lin = nn.Linear(old.in_features, new_out, bias=old.bias is not None)
    with torch.no_grad():
        lin.weight.copy_((_r(old.out_features, new_out, old.weight).T @ old.weight).contiguous())
        if old.bias is not None:
            lin.bias.copy_(_vec(old.bias, old.out_features, new_out))
    return lin


def _linear_in(old: nn.Linear, new_in: int) -> nn.Linear:
    lin = nn.Linear(new_in, old.out_features, bias=old.bias is not None)
    with torch.no_grad():
        lin.weight.copy_((old.weight @ _r(old.in_features, new_in, old.weight)).contiguous())
        if old.bias is not None:
            lin.bias.copy_(old.bias)
    return lin


def _bn_out(old: nn.BatchNorm1d, n: int) -> nn.BatchNorm1d:
    bn = nn.BatchNorm1d(n, eps=old.eps, momentum=old.momentum, affine=old.affine, track_running_stats=old.track_running_stats)
    with torch.no_grad():
        for field in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(old, field, None)
            if t is not None:
                getattr(bn, field).copy_(_vec(t, old.num_features, n))
    return bn


def _passthrough(gm, node):
    if node.op == "call_function":
        return node.target in _ACT_FN
    if node.op == "call_module":
        m = get_layer_module(node.target, gm)
        return isinstance(m, _PASS)
    return False


def _shrink_out(gm, layer_id, mod, new_out):
    if isinstance(mod, nn.Linear):
        _set_module(gm, layer_id, _linear_out(mod, new_out))
    elif isinstance(mod, nn.BatchNorm1d):
        _set_module(gm, layer_id, _bn_out(mod, new_out))
    else:
        raise TypeError(layer_id)


def _is_fork(node):
    return len(node.users) > 1


def _branch_out(gm, node, old_w, new_w, seen, *, nested_add_root=False):
    key = (node.name, old_w, new_w, "out")
    if key in seen:
        return
    seen.add(key)
    if node.op == "call_function" and node.target == operator.add:
        if nested_add_root:
            for inp in node.all_input_nodes:
                _branch_out(gm, inp, old_w, new_w, seen, nested_add_root=False)
        return
    if node.op == "call_module":
        mod = get_layer_module(node.target, gm)
        if isinstance(mod, (nn.Linear, nn.BatchNorm1d)):
            _shrink_out(gm, str(node.target), mod, new_w)
            return
    if _passthrough(gm, node):
        for pred in node.all_input_nodes:
            _branch_out(gm, pred, old_w, new_w, seen, nested_add_root=False)
        return
    raise NotImplementedError(f"unsupported branch node {node.op} {node.target}")


def _branch_in_to_fork(gm, branch_tip, add_node, old_w, new_w, seen):
    """Shrink input width of linears on the branch side between fork and branch_tip."""
    for pred in branch_tip.all_input_nodes:
        _branch_in_visit(gm, pred, add_node, old_w, new_w, seen)


def _branch_in_visit(gm, node, add_node, old_w, new_w, seen):
    if node in add_node.all_input_nodes:
        return
    key = (node.name, old_w, new_w, "in")
    if key in seen:
        return
    seen.add(key)
    for pred in node.all_input_nodes:
        _branch_in_visit(gm, pred, add_node, old_w, new_w, seen)
    if node.op != "call_module":
        return
    mod = get_layer_module(node.target, gm)
    name = str(node.target)
    if isinstance(mod, nn.Linear) and mod.in_features == old_w:
        _set_module(gm, name, _linear_in(mod, new_w))
    elif isinstance(mod, nn.BatchNorm1d) and mod.num_features == old_w:
        _set_module(gm, name, _bn_out(mod, new_w))


def _propagate(gm, node, old_w, new_w, seen):
    key = (node.name, old_w, new_w)
    if key in seen:
        return
    seen.add(key)
    for user in list(node.users):
        if user.op == "output":
            continue
        if user.op == "call_function" and user.target == operator.add:
            for inp in user.all_input_nodes:
                if inp is not node:
                    nested = inp.op == "call_function" and inp.target == operator.add
                    _branch_out(gm, inp, old_w, new_w, seen, nested_add_root=nested)
            if not _is_fork(node):
                _branch_in_to_fork(gm, node, user, old_w, new_w, seen)
            _propagate(gm, user, old_w, new_w, seen)
            continue
        if _passthrough(gm, user):
            _propagate(gm, user, old_w, new_w, seen)
            continue
        if user.op != "call_module":
            raise NotImplementedError(f"unsupported user {user.op} {user.target}")
        mod = get_layer_module(user.target, gm)
        name = str(user.target)
        if isinstance(mod, nn.Linear):
            _set_module(gm, name, _linear_in(mod, new_w))
        elif isinstance(mod, nn.BatchNorm1d):
            _set_module(gm, name, _bn_out(mod, new_w))
            _propagate(gm, user, old_w, new_w, seen)
        else:
            raise NotImplementedError(f"unsupported module {type(mod)}")


def shrink_layer_output(gm, layer_id: str, ratio: float) -> fx.GraphModule:
    gm = _gm(gm)
    mod = get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is not nn.Linear")
    old, new = mod.out_features, max(1, int(mod.out_features * ratio))
    if new >= old:
        return gm
    _set_module(gm, layer_id, _linear_out(mod, new))
    _propagate(gm, _node(gm, layer_id), old, new, set())
    gm.recompile()
    return gm
