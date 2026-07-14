import torch
import torch.fx as fx
import torch.nn as nn
from collections import deque

from growingnn.core import config
from growingnn.core.config import PASSTHROUGH_MODULES_TO_UPDATE, PROPAGATION_RESIZABLE_MODULES, PASSTHROUGH_MODULES
from growingnn.core.logger import logger
from growingnn.utils.fx import ModuleResolver, NodeEditor, NodeTypeChecker, NodeWidthAnalyser
from growingnn.actions.utils.layer_Factory import ConvFactory, LinearFactory
from growingnn.utils.quaziIdentity import get_reshsper


def _find_conv_submodule(mod: nn.Module, prefix: str) -> tuple[str, nn.Conv2d] | None:
    if isinstance(mod, nn.Conv2d):
        return prefix, mod
    if isinstance(mod, nn.Sequential):
        for name, child in mod.named_children():
            path = f"{prefix}.{name}" if prefix else name
            found = _find_conv_submodule(child, path)
            if found is not None:
                return found
    return None


def _sequential_branch_resizable(mod: nn.Module) -> bool:
    if isinstance(mod, PROPAGATION_RESIZABLE_MODULES) or isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
        return True
    if isinstance(mod, PASSTHROUGH_MODULES):
        return False
    if isinstance(mod, nn.Sequential):
        return any(_sequential_branch_resizable(child) for child in mod.children())
    return False


def _rescale_sequential_output(gm, name: str, mod: nn.Sequential, width: int) -> None:
    found = _find_conv_submodule(mod, name)
    if found is None:
        return
    path, conv = found
    if conv.out_channels != width:
        NodeEditor.replace_submodule(gm, path, ConvFactory.create_conv_with_rescaled_output_channels(conv, width))


def _norm_feature_width(mod: nn.Module) -> int:
    if isinstance(mod, nn.LayerNorm):
        return mod.normalized_shape[0]
    if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return mod.num_features
    raise TypeError(f"Unsupported norm module type: {type(mod).__name__}")


def _module_output_width(mod: nn.Module) -> int:
    if isinstance(mod, nn.Linear):
        return mod.out_features
    if isinstance(mod, nn.Conv2d):
        return mod.out_channels
    if isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
        return _norm_feature_width(mod)
    raise TypeError(f"Unsupported module type for width query: {type(mod).__name__}")


def _module_device_dtype(mod: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(mod.parameters(), None)
    if param is not None:
        return param.device, param.dtype
    buf = next(mod.buffers(), None)
    if buf is not None:
        return buf.device, buf.dtype
    return torch.device("cpu"), torch.float32


def _rescale_batch_norm(gm, name, mod, width):
    if mod.num_features == width:
        return
    device, dtype = mod.weight.device, mod.weight.dtype
    bn = type(mod)(width, eps=mod.eps, momentum=mod.momentum, affine=mod.affine, track_running_stats=mod.track_running_stats)
    bn = bn.to(device=device, dtype=dtype)
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


def _rescale_layer_norm(gm, name, mod: nn.LayerNorm, width: int) -> None:
    old_width = mod.normalized_shape[0]
    if old_width == width:
        return
    device, dtype = _module_device_dtype(mod)
    ln = nn.LayerNorm(width, eps=mod.eps, elementwise_affine=mod.elementwise_affine)
    ln = ln.to(device=device, dtype=dtype)
    with torch.no_grad():
        if mod.elementwise_affine:
            R = get_reshsper(old_width, width, dtype=dtype, device=device)
            ln.weight.copy_((R.T @ mod.weight).contiguous())
            ln.bias.copy_((R.T @ mod.bias).contiguous())
    NodeEditor.replace_submodule(gm, name, ln)


def _rescale_norm_output(gm, name: str, mod: nn.Module, width: int) -> None:
    if isinstance(mod, nn.LayerNorm):
        _rescale_layer_norm(gm, name, mod, width)
    else:
        _rescale_batch_norm(gm, name, mod, width)


def _rescale_linear_output(gm, name, mod: nn.Linear, width: int) -> None:
    if mod.out_features != width:
        NodeEditor.replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_neurons(mod, width))


def _rescale_linear_input(gm, name, mod: nn.Linear, width: int) -> None:
    if mod.in_features != width:
        NodeEditor.replace_submodule(gm, name, LinearFactory.create_linear_with_rescaled_connections(mod, width))


def _rescale_conv_output(gm, name, mod: nn.Conv2d, width: int) -> None:
    if mod.out_channels != width:
        NodeEditor.replace_submodule(gm, name, ConvFactory.create_conv_with_rescaled_output_channels(mod, width))


def _rescale_conv_input(gm, name, mod: nn.Conv2d, width: int) -> None:
    if mod.in_channels != width:
        NodeEditor.replace_submodule(gm, name, ConvFactory.create_conv_with_rescaled_input_channels(mod, width))


_OUTPUT_RESIZE_CHAIN = (
    (nn.Linear, _rescale_linear_output),
    (nn.Conv2d, _rescale_conv_output),
    (PASSTHROUGH_MODULES_TO_UPDATE, _rescale_norm_output),
)
_INPUT_RESIZE_CHAIN = (
    (nn.Linear, _rescale_linear_input),
    (nn.Conv2d, _rescale_conv_input),
    (PASSTHROUGH_MODULES_TO_UPDATE, _rescale_norm_output),
)


def _apply_output_resize(gm, name: str, mod: nn.Module, width: int) -> bool:
    for types, handler in _OUTPUT_RESIZE_CHAIN:
        if isinstance(mod, types):
            handler(gm, name, mod, width)
            return True
    return False


def _apply_input_resize(gm, name: str, mod: nn.Module, width: int) -> bool:
    for types, handler in _INPUT_RESIZE_CHAIN:
        if isinstance(mod, types):
            handler(gm, name, mod, width)
            return True
    return False


def _rescale_output_neurons(gm, name, mod, width):
    _apply_output_resize(gm, name, mod, width)


def _rescale_input_connections(gm, name, mod, width):
    if not NodeWidthAnalyser.all_sites_match_width(gm, name, width):
        return
    _apply_input_resize(gm, name, mod, width)


def _is_square_resizable(mod: nn.Module) -> bool:
    return (
        (isinstance(mod, nn.Linear) and mod.in_features == mod.out_features)
        or (isinstance(mod, nn.Conv2d) and mod.in_channels == mod.out_channels)
    )


def _fork_blocks_propagation(gm, node: fx.Node, width: int) -> bool:
    if not NodeTypeChecker.is_fork(node):
        return False
    actual = NodeWidthAnalyser.node_output_width(gm, node)
    return actual is not None and actual != width


def _sync_add_siblings_backward(gm, node, width, seen, *, via_pass=False, at_add=None):
    """Walk backward from add-node sibling branches to align their output width."""
    logger.debug("sync_add_siblings_backward: %s", node.name)
    key = ("s", node.name, width)
    if key in seen:
        return
    seen.add(key)
    if NodeTypeChecker.is_add(node):
        if not via_pass:
            for inp in node.all_input_nodes:
                _sync_add_siblings_backward(gm, inp, width, seen, at_add=node)
        return
    if node.op == "call_module":
        mod = ModuleResolver.get_layer_module(node.target, gm)
        if isinstance(mod, PROPAGATION_RESIZABLE_MODULES):
            if NodeTypeChecker.is_fork(node) and (at_add is None or node not in at_add.all_input_nodes):
                return
            _rescale_output_neurons(gm, str(node.target), mod, width)
            propagate_neuron_change(gm, node, width, seen)
            return
        if isinstance(mod, nn.Sequential) and _sequential_branch_resizable(mod):
            _rescale_sequential_output(gm, str(node.target), mod, width)
            for pred in node.all_input_nodes:
                _sync_add_siblings_backward(gm, pred, width, seen, via_pass=True)
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
            ):
                continue
            _sync_add_siblings_backward(gm, pred, width, seen, via_pass=True)


def _align_inputs_backward(gm, node, add_node, width, seen):
    """Walk backward through predecessors to rescale their input features."""
    if node in add_node.all_input_nodes or NodeTypeChecker.is_fork(node):
        return
    key = ("b", node.name, width)
    if key in seen:
        return
    seen.add(key)
    for pred in node.all_input_nodes:
        _align_inputs_backward(gm, pred, add_node, width, seen)
    if node.op == "call_module" and NodeWidthAnalyser.inputs_match_width(gm, node, width):
        _rescale_input_connections(gm, str(node.target), ModuleResolver.get_layer_module(node.target, gm), width)


def _prepare_add_node(gm, source: fx.Node, add_node: fx.Node, width: int, seen: set) -> None:
    for inp in add_node.all_input_nodes:
        if inp is not source:
            _sync_add_siblings_backward(gm, inp, width, seen, at_add=add_node)
    if not NodeTypeChecker.is_fork(source):
        for pred in source.all_input_nodes:
            _align_inputs_backward(gm, pred, add_node, width, seen)


def _propagate_through_resizable(gm, source: fx.Node, user: fx.Node, width: int, seen: set, queue: deque) -> None:
    mod = ModuleResolver.get_layer_module(user.target, gm)
    if mod is None or not isinstance(mod, PROPAGATION_RESIZABLE_MODULES):
        return
    name = str(user.target)
    if not NodeWidthAnalyser.inputs_match_width(gm, user, width):
        logger.debug("propagate_neuron_change --- skip input width mismatch: %s", name)
        return
    was_square = _is_square_resizable(mod)
    _rescale_input_connections(gm, name, mod, width)
    updated = ModuleResolver.get_layer_module(name, gm)
    out_w = _module_output_width(updated)
    if NodeTypeChecker.is_add(source) and was_square and out_w != width:
        _rescale_output_neurons(gm, name, updated, width)
        out_w = width
    queue.append((user, out_w))


def _propagate_edge(gm, source: fx.Node, user: fx.Node, width: int, seen: set, queue: deque) -> None:
    if NodeTypeChecker.is_add(user):
        _prepare_add_node(gm, source, user, width, seen)
        queue.append((user, width))
        return
    if NodeTypeChecker.is_passthrough(gm, user):
        queue.append((user, width))
        return
    if user.op == "call_module":
        mod = ModuleResolver.get_layer_module(user.target, gm)
        if mod is None:
            logger.debug("propagate_neuron_change --- skip missing module: %s", user.target)
            return
        if isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
            _rescale_output_neurons(gm, str(user.target), mod, width)
            queue.append((user, width))
            return
        if isinstance(mod, PROPAGATION_RESIZABLE_MODULES):
            _propagate_through_resizable(gm, source, user, width, seen, queue)
            return
        logger.debug("propagate_neuron_change --- skip non-resizable module: %s", user.target)
        return
    logger.debug("propagate_neuron_change --- skip non-call_module: %s op=%s", user.name, user.op)


def _within_linear_matrix_limit(mod: nn.Linear, new_out: int) -> bool:
    max_side = max(mod.out_features, new_out)
    return (
        max_side * max_side <= config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE
        and mod.in_features * new_out <= config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE
    )


def can_resize_linear_output(
    gm: nn.Module | fx.GraphModule,
    layer_id: str,
    new_width: int,
) -> bool:
    """Return True when a Linear layer output can be rescaled to new_width and propagated."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear) or new_width == mod.out_features:
        return False
    if new_width < mod.out_features:
        if new_width < config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
            return False
    elif not _within_linear_matrix_limit(mod, new_width):
        return False
    node = ModuleResolver.find_call_module(gm.graph.nodes, layer_id)
    return not NodeWidthAnalyser.propagation_hits_unsizable(gm, node)


def resize_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, new_width: int) -> fx.GraphModule:
    """Resize a Linear layer's output to new_width and propagate the change through the graph."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is {type(mod).__name__}, not nn.Linear")
    NodeEditor.replace_submodule(gm, layer_id, LinearFactory.create_linear_with_rescaled_neurons(mod, new_width))
    propagate_neuron_change(gm, ModuleResolver.find_call_module(gm.graph.nodes, layer_id), new_width, set())
    gm.recompile()
    for tensor in list(gm.parameters()) + list(gm.buffers()):
        if tensor.numel() > 0 and not tensor.is_contiguous():
            tensor.data = tensor.data.contiguous()
    return gm


def propagate_neuron_change(gm, node, width, seen):
    """Forward-propagate width through consumers using a work queue."""
    queue = deque([(node, width)])
    while queue:
        cur, w = queue.popleft()
        key = ("p", cur.name, w)
        if key in seen:
            continue
        seen.add(key)
        logger.debug("propagate_neuron_change: %s width=%s", cur.name, w)
        if _fork_blocks_propagation(gm, cur, w):
            continue
        for user in list(cur.users):
            if user.op == "output":
                logger.debug("propagate_neuron_change --- skip output: %s", user.name)
                continue
            _propagate_edge(gm, cur, user, w, seen, queue)
