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


def _tensor_width_at(gm, node, cache: dict) -> int | None:
    """Channel width at node output; None if unknown."""
    if node in cache:
        return cache[node]
    if node.op == "call_module":
        mod = get_layer_module(node.target, gm)
        if isinstance(mod, nn.Linear):
            cache[node] = mod.out_features
            return mod.out_features
        if isinstance(mod, nn.BatchNorm1d):
            cache[node] = mod.num_features
            return mod.num_features
    if _passthrough(gm, node) and node.all_input_nodes:
        w = _tensor_width_at(gm, node.all_input_nodes[0], cache)
        cache[node] = w
        return w
    if node.op == "call_function" and node.target == operator.add and node.all_input_nodes:
        w = _tensor_width_at(gm, node.all_input_nodes[0], cache)
        cache[node] = w
        return w
    cache[node] = None
    return None


def _linear_inputs_match_width(gm, linear_node, width, cache: dict) -> bool:
    if not linear_node.all_input_nodes:
        return False
    return all(_tensor_width_at(gm, inp, cache) == width for inp in linear_node.all_input_nodes)


def _module_all_uses_match_input_width(gm, module_name, width, cache: dict) -> bool:
    """True when every FX call site for module_name sees input width ``width``."""
    for n in gm.graph.nodes:
        if n.op != "call_module" or n.target != module_name:
            continue
        if not _linear_inputs_match_width(gm, n, width, cache):
            return False
    return True


def _maybe_narrow_linear_in(gm, name, mod, width, cache: dict) -> nn.Linear:
    if mod.in_features == width:
        return mod
    if not _module_all_uses_match_input_width(gm, name, width, cache):
        return mod
    return _linear_in(mod, width)


def _sync_branch_to_width(gm, node, target_w, seen, *, from_passthrough=False, at_add=None):
    """Shrink a sibling branch output to target_w, then propagate from that branch tip."""
    key = ("branch", node.name, target_w)
    if key in seen:
        return
    seen.add(key)
    if node.op == "call_function" and node.target == operator.add:
        if from_passthrough:
            return
        for inp in node.all_input_nodes:
            _sync_branch_to_width(gm, inp, target_w, seen, from_passthrough=False, at_add=node)
        return
    if node.op == "call_module":
        mod = get_layer_module(node.target, gm)
        name = str(node.target)
        if isinstance(mod, nn.Linear):
            if _is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            if mod.out_features > target_w:
                _shrink_out(gm, name, mod, target_w)
            _propagate_width(gm, node, target_w, seen)
            return
        if isinstance(mod, nn.BatchNorm1d):
            if _is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            if mod.num_features > target_w:
                _shrink_out(gm, name, mod, target_w)
            _propagate_width(gm, node, target_w, seen)
            return
    if _passthrough(gm, node):
        for pred in node.all_input_nodes:
            if _is_fork(pred) and not _passthrough(gm, pred):
                continue
            _sync_branch_to_width(gm, pred, target_w, seen, from_passthrough=True)
        return
    if _is_fork(node):
        return
    raise NotImplementedError(f"unsupported branch node {node.op} {node.target}")


def _branch_in_to_fork(gm, branch_tip, add_node, target_w, seen):
    for pred in branch_tip.all_input_nodes:
        _branch_in_visit(gm, pred, add_node, target_w, seen)


def _branch_in_visit(gm, node, add_node, target_w, seen):
    if node in add_node.all_input_nodes or _is_fork(node):
        return
    key = ("in", node.name, target_w)
    if key in seen:
        return
    seen.add(key)
    for pred in node.all_input_nodes:
        _branch_in_visit(gm, pred, add_node, target_w, seen)
    if node.op != "call_module":
        return
    cache: dict = {}
    if not _linear_inputs_match_width(gm, node, target_w, cache):
        return
    mod = get_layer_module(node.target, gm)
    name = str(node.target)
    if isinstance(mod, nn.Linear) and mod.in_features > target_w:
        _set_module(gm, name, _maybe_narrow_linear_in(gm, name, mod, target_w, cache))
    elif isinstance(mod, nn.BatchNorm1d) and mod.num_features > target_w:
        _set_module(gm, name, _bn_out(mod, target_w))


def _propagate_width(gm, node, width, seen):
    """Propagate channel width downstream from node (tensor width is `width`)."""
    key = ("prop", node.name, width)
    if key in seen:
        return
    seen.add(key)
    cache: dict = {}
    if _is_fork(node) and _tensor_width_at(gm, node, cache) != width:
        return
    for user in list(node.users):
        if user.op == "output":
            continue
        if user.op == "call_function" and user.target == operator.add:
            for inp in user.all_input_nodes:
                if inp is not node:
                    _sync_branch_to_width(gm, inp, width, seen, from_passthrough=False, at_add=user)
            if not _is_fork(node):
                _branch_in_to_fork(gm, node, user, width, seen)
            _propagate_width(gm, user, width, seen)
            continue
        if _passthrough(gm, user):
            _propagate_width(gm, user, width, seen)
            continue
        if user.op != "call_module":
            raise NotImplementedError(f"unsupported user {user.op} {user.target}")
        mod = get_layer_module(user.target, gm)
        name = str(user.target)
        if isinstance(mod, nn.Linear):
            if not _linear_inputs_match_width(gm, user, width, cache):
                continue
            narrowed = _maybe_narrow_linear_in(gm, name, mod, width, cache)
            if narrowed is not mod:
                _set_module(gm, name, narrowed)
                mod = narrowed
            _propagate_width(gm, user, mod.out_features, seen)
        elif isinstance(mod, nn.BatchNorm1d):
            if mod.num_features != width:
                _set_module(gm, name, _bn_out(mod, width))
            _propagate_width(gm, user, width, seen)
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
    _propagate_width(gm, _node(gm, layer_id), new, set())
    gm.recompile()
    return gm
