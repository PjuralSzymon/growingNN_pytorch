import torch.nn as nn

from growingnn.actions.utils.layer_analyser import (
    is_passthrough, is_fork, is_add,
    node_output_width, inputs_match_width, all_sites_match_width,
)
from growingnn.actions.utils.model_analyser import get_layer_module
from growingnn.actions.utils.model_transformations import replace_submodule
from growingnn.actions.utils.layer_Factory import LinearFactory


def _rescale_output_neurons(gm, name, mod, width):
    """Replace a Linear module with one that has fewer output neurons."""
    if isinstance(mod, nn.Linear) and mod.out_features > width:
        replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_neurons(mod, width))


def _rescale_input_connections(gm, name, mod, width):
    """Replace a Linear module with one whose in_features matches the new width."""
    if isinstance(mod, nn.Linear) and mod.in_features != width and all_sites_match_width(gm, name, width):
        replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_connections(mod, width))


# --------------- graph traversal ---------------

def _sync_add_siblings_backward(gm, node, width, seen, *, via_pass=False, at_add=None):
    """Walk backward from add-node sibling branches to align their output width."""
    key = ("s", node.name, width)
    if key in seen: return
    seen.add(key)
    if is_add(node):
        if not via_pass:
            for inp in node.all_input_nodes:
                _sync_add_siblings_backward(gm, inp, width, seen, at_add=node)
        return
    if node.op == "call_module":
        mod = get_layer_module(node.target, gm)
        if isinstance(mod, nn.Linear):
            if is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            _rescale_output_neurons(gm, str(node.target), mod, width)
            propagate_neuron_change(gm, node, width, seen)
            return
    if is_passthrough(gm, node):
        for pred in node.all_input_nodes:
            if is_fork(pred) and not is_passthrough(gm, pred): continue
            _sync_add_siblings_backward(gm, pred, width, seen, via_pass=True)
        return
    if is_fork(node): return
    if node.op == "call_module":
        return


def _align_inputs_backward(gm, node, add_node, width, seen):
    """Walk backward through predecessors to rescale their input features."""
    if node in add_node.all_input_nodes or is_fork(node): return
    key = ("b", node.name, width)
    if key in seen: return
    seen.add(key)
    for pred in node.all_input_nodes:
        _align_inputs_backward(gm, pred, add_node, width, seen)
    if node.op == "call_module" and inputs_match_width(gm, node, width):
        _rescale_input_connections(gm, str(node.target), get_layer_module(node.target, gm), width)


def propagate_neuron_change(gm, node, width, seen):
    """Walk forward through the graph and resize every downstream layer to match width."""
    key = ("p", node.name, width)
    if key in seen: return
    seen.add(key)
    if is_fork(node) and node_output_width(gm, node) != width: return
    for user in list(node.users):
        if user.op == "output": continue
        if is_add(user):
            for inp in user.all_input_nodes:
                if inp is not node:
                    _sync_add_siblings_backward(gm, inp, width, seen, at_add=user)
            if not is_fork(node):
                for pred in node.all_input_nodes:
                    _align_inputs_backward(gm, pred, user, width, seen)
            propagate_neuron_change(gm, user, width, seen)
            continue
        if is_passthrough(gm, user):
            propagate_neuron_change(gm, user, width, seen)
            continue
        if user.op != "call_module":
            continue
        mod = get_layer_module(user.target, gm)
        if mod is None:
            continue
        name = str(user.target)
        if isinstance(mod, nn.Linear):
            if not inputs_match_width(gm, user, width): continue
            _rescale_input_connections(gm, name, mod, width)
            propagate_neuron_change(gm, user, get_layer_module(name, gm).out_features, seen)
