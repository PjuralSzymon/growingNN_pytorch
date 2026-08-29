import torch
import torch.fx as fx
import torch.nn as nn

from growingnn.core import config
from growingnn.core.config import PASSTHROUGH_MODULES_TO_UPDATE, PROPAGATION_RESIZABLE_MODULES, PASSTHROUGH_MODULES
from growingnn.core.logger import logger
from growingnn.utils.fx import (
    LayerShapeAnalyser,
    ModuleResolver,
    NodeEditor,
    NodeTypeChecker,
    NodeWidthAnalyser,
)
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


def _norm_feature_width(mod: nn.Module) -> int | None:
    if isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
        return int(mod.num_features)
    if isinstance(mod, nn.LayerNorm):
        shape = mod.normalized_shape
        if isinstance(shape, tuple):
            return int(shape[-1]) if shape else None
        return int(shape)
    return None


def _module_output_width(mod: nn.Module) -> int:
    if isinstance(mod, nn.Linear):
        return mod.out_features
    if isinstance(mod, nn.Conv2d):
        return mod.out_channels
    norm_w = _norm_feature_width(mod)
    if norm_w is not None:
        return norm_w
    raise TypeError(f"Unsupported module type for width query: {type(mod).__name__}")


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
    if _norm_feature_width(mod) == width:
        return
    device, dtype = mod.weight.device, mod.weight.dtype
    # Keep trailing-dim LayerNorm (ViT-style); drop leading normalized dims if any.
    ln = nn.LayerNorm(width, eps=mod.eps, elementwise_affine=mod.elementwise_affine)
    ln = ln.to(device=device, dtype=dtype)
    with torch.no_grad():
        if mod.elementwise_affine:
            old_w = _norm_feature_width(mod)
            R = get_reshsper(old_w, width, dtype=mod.weight.dtype, device=mod.weight.device)
            ln.weight.copy_((R.T @ mod.weight.reshape(-1)).contiguous())
            ln.bias.copy_((R.T @ mod.bias.reshape(-1)).contiguous())
    NodeEditor.replace_submodule(gm, name, ln)


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
    (PASSTHROUGH_MODULES_TO_UPDATE, _rescale_batch_norm),
    (nn.LayerNorm, _rescale_layer_norm),
)
_INPUT_RESIZE_CHAIN = (
    (nn.Linear, _rescale_linear_input),
    (nn.Conv2d, _rescale_conv_input),
    (PASSTHROUGH_MODULES_TO_UPDATE, _rescale_batch_norm),
    (nn.LayerNorm, _rescale_layer_norm),
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


_HEAD_NAME_CANDIDATES = ("output", "head", "fc", "classifier")


def _find_head_linear_name(gm: fx.GraphModule) -> str | None:
    """Prefer a named classifier Linear; else a Linear reached only through passthrough/norm."""
    for name in _HEAD_NAME_CANDIDATES:
        mod = ModuleResolver.get_layer_module(name, gm)
        if isinstance(mod, nn.Linear):
            return name
    output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
    if output_node is None or not output_node.all_input_nodes:
        return None
    cur = output_node.all_input_nodes[0]
    seen: set[fx.Node] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        if cur.op == "call_module":
            mod = ModuleResolver.get_layer_module(cur.target, gm)
            if isinstance(mod, nn.Linear):
                return str(cur.target)
            if isinstance(mod, (PASSTHROUGH_MODULES, PASSTHROUGH_MODULES_TO_UPDATE, nn.LayerNorm)):
                cur = cur.all_input_nodes[0] if cur.all_input_nodes else None
                continue
            return None
        if NodeTypeChecker.is_passthrough(gm, cur):
            cur = cur.all_input_nodes[0] if cur.all_input_nodes else None
            continue
        return None
    return None


def _snapshot_pinned_head_out(gm: fx.GraphModule) -> tuple[str | None, int | None]:
    """Return (head_name, out_features) to keep fixed during the global fix sweep."""
    name = _find_head_linear_name(gm)
    if name is None:
        return None, None
    mod = ModuleResolver.get_layer_module(name, gm)
    if not isinstance(mod, nn.Linear):
        return None, None
    return name, int(mod.out_features)


def _module_input_width(mod: nn.Module) -> int | None:
    if isinstance(mod, nn.Linear):
        return mod.in_features
    if isinstance(mod, nn.Conv2d):
        return mod.in_channels
    return _norm_feature_width(mod)


def _agreed_input_width(gm: fx.GraphModule, node: fx.Node) -> int | None:
    if not node.all_input_nodes:
        return None
    widths = [NodeWidthAnalyser.node_output_width(gm, inp) for inp in node.all_input_nodes]
    if any(w is None for w in widths) or len(set(widths)) != 1:
        return None
    return widths[0]


def _nearest_output_resize_target(
    gm: fx.GraphModule, node: fx.Node
) -> tuple[str, nn.Module] | None:
    """Walk upstream through passthrough/BN/add to the nearest Linear/Conv/Sequential to rescale."""
    seen: set[fx.Node] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "call_module":
            mod = ModuleResolver.get_layer_module(cur.target, gm)
            if mod is None:
                continue
            name = str(cur.target)
            if isinstance(mod, PROPAGATION_RESIZABLE_MODULES):
                return name, mod
            if isinstance(mod, nn.Sequential) and _sequential_branch_resizable(mod):
                return name, mod
            # BN / LayerNorm follow the producer; do not stop here or add-sync oscillates.
            if isinstance(mod, (PASSTHROUGH_MODULES, PASSTHROUGH_MODULES_TO_UPDATE, nn.LayerNorm)):
                stack.extend(cur.all_input_nodes)
                continue
        if NodeTypeChecker.is_passthrough(gm, cur) or NodeTypeChecker.is_add(cur):
            stack.extend(cur.all_input_nodes)
    return None


def _current_site_output_width(gm: fx.GraphModule, name: str, mod: nn.Module) -> int | None:
    if isinstance(mod, nn.Sequential):
        found = _find_conv_submodule(mod, name)
        return None if found is None else found[1].out_channels
    if isinstance(mod, (PROPAGATION_RESIZABLE_MODULES, PASSTHROUGH_MODULES_TO_UPDATE)):
        return _module_output_width(mod)
    return None


def _fix_site_output_width(gm: fx.GraphModule, name: str, mod: nn.Module, width: int) -> bool:
    before = _current_site_output_width(gm, name, mod)
    if before == width:
        return False
    if isinstance(mod, nn.Sequential):
        _rescale_sequential_output(gm, name, mod, width)
    else:
        _rescale_output_neurons(gm, name, mod, width)
    updated = ModuleResolver.get_layer_module(name, gm)
    if updated is None:
        return before != width
    after = _current_site_output_width(gm, name, updated)
    return after != before


def _fix_module_input_mismatch(gm: fx.GraphModule, node: fx.Node) -> bool:
    if node.op != "call_module":
        return False
    mod = ModuleResolver.get_layer_module(node.target, gm)
    if mod is None:
        return False
    name = str(node.target)
    target_w = _agreed_input_width(gm, node)
    if target_w is None:
        return False
    current = _module_input_width(mod)
    if current is None or current == target_w:
        return False
    if isinstance(mod, (PROPAGATION_RESIZABLE_MODULES, PASSTHROUGH_MODULES_TO_UPDATE, nn.LayerNorm)):
        _rescale_input_connections(gm, name, mod, target_w)
        updated = ModuleResolver.get_layer_module(name, gm)
        return updated is not None and _module_input_width(updated) == target_w and current != target_w
    return False


def _fix_add_input_mismatch(
    gm: fx.GraphModule,
    node: fx.Node,
    *,
    align_add_to: int,
    pinned_head_name: str | None,
) -> bool:
    if not NodeTypeChecker.is_add(node):
        return False
    widths: list[int] = []
    for inp in node.all_input_nodes:
        w = NodeWidthAnalyser.node_output_width(gm, inp)
        if w is None:
            return False
        widths.append(w)
    if len(set(widths)) <= 1:
        return False
    target = align_add_to if align_add_to in widths else min(widths)
    changed = False
    for inp in node.all_input_nodes:
        site = _nearest_output_resize_target(gm, inp)
        if site is None:
            continue
        name, mod = site
        if pinned_head_name is not None and name == pinned_head_name:
            continue
        if _fix_site_output_width(gm, name, mod, target):
            changed = True
    return changed


def _fix_norm_mismatch(gm: fx.GraphModule, node: fx.Node) -> bool:
    if node.op != "call_module":
        return False
    mod = ModuleResolver.get_layer_module(node.target, gm)
    if not isinstance(mod, (PASSTHROUGH_MODULES_TO_UPDATE, nn.LayerNorm)):
        return False
    target_w = _agreed_input_width(gm, node)
    current = _module_input_width(mod)
    if target_w is None or current is None or current == target_w:
        return False
    name = str(node.target)
    _rescale_input_connections(gm, name, mod, target_w)
    updated = ModuleResolver.get_layer_module(name, gm)
    return updated is not None and _module_input_width(updated) == target_w


def _pin_head_output(gm: fx.GraphModule, head_name: str | None, pinned_head_out: int | None) -> bool:
    if head_name is None or pinned_head_out is None:
        return False
    mod = ModuleResolver.get_layer_module(head_name, gm)
    if not isinstance(mod, nn.Linear) or mod.out_features == pinned_head_out:
        return False
    _rescale_linear_output(gm, head_name, mod, pinned_head_out)
    updated = ModuleResolver.get_layer_module(head_name, gm)
    return isinstance(updated, nn.Linear) and updated.out_features == pinned_head_out


def _graph_has_width_mismatch(gm: fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if NodeTypeChecker.is_add(node):
            widths = [NodeWidthAnalyser.node_output_width(gm, inp) for inp in node.all_input_nodes]
            if widths and all(w is not None for w in widths) and len(set(widths)) > 1:
                return True
            continue
        if node.op != "call_module":
            continue
        mod = ModuleResolver.get_layer_module(node.target, gm)
        if mod is None:
            continue
        current = _module_input_width(mod)
        if current is None:
            continue
        target_w = _agreed_input_width(gm, node)
        if target_w is not None and current != target_w:
            return True
    return False


def _graph_has_flatten(gm: fx.GraphModule) -> bool:
    return any(NodeTypeChecker.is_flatten_node(node, gm) for node in gm.graph.nodes)


def _refresh_flatten_shape_meta(gm: fx.GraphModule, input_shape: tuple[int, ...] | None) -> None:
    """Run ShapeProp on a still-valid graph so spatial flatten exposes C*H*W in FX meta."""
    if input_shape is None or not _graph_has_flatten(gm):
        return
    LayerShapeAnalyser.collect_layer_shapes(gm, input_shape=input_shape)


def fix_graph_widths(
    gm: fx.GraphModule,
    *,
    align_add_to: int,
    pinned_head_out: int | None,
    pinned_head_name: str | None = None,
    max_passes: int = 32,
) -> None:
    """
    Sequentially sweep the FX graph and fix width mismatches until stable.

    Reuses existing rescale helpers. Does not recurse from the edited layer.
    """
    head_name = pinned_head_name
    if head_name is None and pinned_head_out is not None:
        head_name, _ = _snapshot_pinned_head_out(gm)

    for pass_idx in range(max_passes):
        changed = False
        for node in list(gm.graph.nodes):
            if _fix_module_input_mismatch(gm, node):
                changed = True
            if _fix_add_input_mismatch(
                gm, node, align_add_to=align_add_to, pinned_head_name=head_name
            ):
                changed = True
            if _fix_norm_mismatch(gm, node):
                changed = True
            if _pin_head_output(gm, head_name, pinned_head_out):
                changed = True
        logger.debug("fix_graph_widths pass=%s changed=%s", pass_idx, changed)
        if not changed:
            break
    else:
        logger.error("fix_graph_widths: exceeded max_passes=%s", max_passes)

    if _graph_has_width_mismatch(gm):
        raise RuntimeError(
            f"fix_graph_widths failed to resolve width mismatches after {max_passes} passes"
        )


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


def resize_layer_output(
    gm: nn.Module | fx.GraphModule,
    layer_id: str,
    new_width: int,
    input_shape: tuple[int, ...] | None = None,
) -> fx.GraphModule:
    """Resize a Linear layer's output to new_width, then fix all graph width mismatches."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is {type(mod).__name__}, not nn.Linear")
    head_name, pinned_head_out = _snapshot_pinned_head_out(gm)
    _refresh_flatten_shape_meta(gm, input_shape)
    NodeEditor.replace_submodule(gm, layer_id, LinearFactory.create_linear_with_rescaled_neurons(mod, new_width))
    fix_graph_widths(
        gm,
        align_add_to=new_width,
        pinned_head_out=pinned_head_out,
        pinned_head_name=head_name,
    )
    gm.recompile()
    for tensor in list(gm.parameters()) + list(gm.buffers()):
        if tensor.numel() > 0 and not tensor.is_contiguous():
            tensor.data = tensor.data.contiguous()
    return gm
