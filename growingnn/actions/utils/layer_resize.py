import torch
import torch.nn as nn

from growingnn.core.config import PASSTHROUGH_MODULES_TO_UPDATE
from growingnn.core.logger import logger
from growingnn.utils.fx import ModuleResolver, NodeEditor, NodeTypeChecker, NodeWidthAnalyser
from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.utils.quaziIdentity import get_reshsper


def _rescale_batch_norm(gm, name, mod, width):
    if mod.num_features <= width:
        return
    bn = type(mod)(width, eps=mod.eps, momentum=mod.momentum, affine=mod.affine, track_running_stats=mod.track_running_stats)
    with torch.no_grad():
        if mod.affine:
            R = get_reshsper(mod.num_features, width, dtype=mod.weight.dtype, device=mod.weight.device)
            bn.weight.copy_((R.T @ mod.weight).contiguous())
            bn.bias.copy_((R.T @ mod.bias).contiguous())
        if mod.track_running_stats:
            R = get_reshsper(mod.num_features, width, dtype=mod.running_mean.dtype, device=mod.running_mean.device)
            bn.running_mean.copy_((R.T @ mod.running_mean).contiguous())
            bn.running_var.copy_((R.T @ mod.running_var).contiguous())
            bn.num_batches_tracked.copy_(mod.num_batches_tracked)
    NodeEditor.replace_submodule(gm, name, bn)


# (module type(s), handler) — add a row for new non-Linear layers
_UPDATE = ((PASSTHROUGH_MODULES_TO_UPDATE, _rescale_batch_norm),)


def _rescale_output_neurons(gm, name, mod, width):
    """Replace a module with one that has fewer output features."""
    if isinstance(mod, nn.Linear) and mod.out_features > width:
        NodeEditor.replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_neurons(mod, width))
        return
    for types, fn in _UPDATE:
        if isinstance(mod, types):
            fn(gm, name, mod, width)
            return


def _rescale_input_connections(gm, name, mod, width):
    """Replace a module whose input features match the new width."""
    if isinstance(mod, nn.Linear) and mod.in_features != width and NodeWidthAnalyser.all_sites_match_width(gm, name, width):
        NodeEditor.replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_connections(mod, width))
        return
    for types, fn in _UPDATE:
        if isinstance(mod, types):
            fn(gm, name, mod, width)
            return


# --------------- graph traversal ---------------

def _sync_add_siblings_backward(gm, node, width, seen, *, via_pass=False, at_add=None):
    """Walk backward from add-node sibling branches to align their output width."""
    logger.debug("sync_add_siblings_backward: %s", node.name)
    key = ("s", node.name, width)
    if key in seen: return
    seen.add(key)
    if NodeTypeChecker.is_add(node):
        if not via_pass:
            for inp in node.all_input_nodes:
                _sync_add_siblings_backward(gm, inp, width, seen, at_add=node)
        return
    if node.op == "call_module":
        mod = ModuleResolver.get_layer_module(node.target, gm)
        if isinstance(mod, nn.Linear):
            if NodeTypeChecker.is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            if at_add is not None and NodeWidthAnalyser.fork_shrink_blocked_by_conv_add(gm, node, at_add):
                return
            _rescale_output_neurons(gm, str(node.target), mod, width)
            propagate_neuron_change(gm, node, width, seen)
            return
        if isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
            _rescale_output_neurons(gm, str(node.target), mod, width)
            for pred in node.all_input_nodes:
                _sync_add_siblings_backward(gm, pred, width, seen, via_pass=True)
            return
    if NodeTypeChecker.is_passthrough(gm, node):
        for pred in node.all_input_nodes:
            if NodeTypeChecker.is_fork(pred) and not (
                NodeTypeChecker.is_passthrough(gm, pred)
                or (pred.op == "call_module" and isinstance(ModuleResolver.get_layer_module(pred.target, gm), PASSTHROUGH_MODULES_TO_UPDATE))
            ): continue
            _sync_add_siblings_backward(gm, pred, width, seen, via_pass=True)

def _align_inputs_backward(gm, node, add_node, width, seen):
    """Walk backward through predecessors to rescale their input features."""
    if node in add_node.all_input_nodes or NodeTypeChecker.is_fork(node): return
    key = ("b", node.name, width)
    if key in seen: return
    seen.add(key)
    for pred in node.all_input_nodes:
        _align_inputs_backward(gm, pred, add_node, width, seen)
    if node.op == "call_module" and NodeWidthAnalyser.inputs_match_width(gm, node, width):
        _rescale_input_connections(gm, str(node.target), ModuleResolver.get_layer_module(node.target, gm), width)


def propagate_neuron_change(gm, node, width, seen):
    """Walk forward through the graph and resize every downstream layer to match width."""
    key = ("p", node.name, width)
    logger.debug("propagate_neuron_change: %s", node.name)
    if key in seen: return
    seen.add(key)
    if NodeTypeChecker.is_fork(node) and NodeWidthAnalyser.node_output_width(gm, node) != width: return
    for user in list(node.users):
        if user.op == "output":
            logger.debug("propagate_neuron_change --- skip output: %s", user.name)
            continue
        if NodeTypeChecker.is_add(user):
            for inp in user.all_input_nodes:
                if inp is not node:
                    _sync_add_siblings_backward(gm, inp, width, seen, at_add=user)
            if not NodeTypeChecker.is_fork(node):
                for pred in node.all_input_nodes:
                    _align_inputs_backward(gm, pred, user, width, seen)
            propagate_neuron_change(gm, user, width, seen)
            continue
        if user.op == "call_module" and isinstance(ModuleResolver.get_layer_module(user.target, gm), PASSTHROUGH_MODULES_TO_UPDATE):
            mod = ModuleResolver.get_layer_module(user.target, gm)
            _rescale_output_neurons(gm, str(user.target), mod, width)
            propagate_neuron_change(gm, user, width, seen)
            continue
        if NodeTypeChecker.is_passthrough(gm, user):
            propagate_neuron_change(gm, user, width, seen)
            continue
        if user.op != "call_module":
            logger.debug("propagate_neuron_change --- skip non-call_module: %s op=%s", user.name, user.op)
            continue
        mod = ModuleResolver.get_layer_module(user.target, gm)
        if mod is None:
            logger.debug("propagate_neuron_change --- skip missing module: %s", user.target)
            continue
        name = str(user.target)
        if isinstance(mod, nn.Linear):
            if not NodeWidthAnalyser.inputs_match_width(gm, user, width):
                logger.debug("propagate_neuron_change --- skip input width mismatch: %s", name)
                continue
            _rescale_input_connections(gm, name, mod, width)
            propagate_neuron_change(gm, user, ModuleResolver.get_layer_module(name, gm).out_features, seen)
        else:
            logger.debug("propagate_neuron_change --- skip non-linear: %s", name)
