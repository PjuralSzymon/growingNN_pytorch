"""Architecture-neutral descriptions of common FX computation operations.

Descriptors explain what an FX node computes and where its parameters come
from. They are deliberately separate from candidate policy and graph edits:
recognizing a projection does not by itself mean its dimensions can be changed
safely across the surrounding graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearOperationDescriptor:
    """One module-backed or functional linear projection in an FX graph."""

    node: fx.Node
    form: str
    input_node: fx.Node | None
    weight_node: fx.Node | None
    bias_node: fx.Node | None
    weight_target: str | None
    parameter_aliases: tuple[str, ...]
    in_features: int | None
    out_features: int | None
    weight_orientation: str
    recognized: bool
    resize_support: str
    diagnostic: str | None = None


def is_functional_convolution_operation(node: fx.Node) -> bool:
    """Recognize visible functional/ATen Conv1d, Conv2d, and Conv3d nodes.

    This is an operator-family check for graph-boundary discovery. It does not
    infer tensor layout or approve a convolution mutation; callers still need
    an explicit feature-axis interface policy.
    """
    if node.op != "call_function":
        return False
    target = node.target
    functional_targets = {
        F.conv1d, F.conv2d, F.conv3d,
        getattr(torch, "conv1d", None),
        getattr(torch, "conv2d", None),
        getattr(torch, "conv3d", None),
    }
    if target in functional_targets:
        return True
    name = str(target).lower()
    return any(
        marker in name
        for marker in ("aten.conv1d.", "aten.conv2d.", "aten.conv3d.", "aten.convolution.")
    )


def _resolve_attr(gm: fx.GraphModule, target: str) -> Any:
    try:
        return gm.get_parameter(target)
    except (AttributeError, TypeError):
        pass
    try:
        return gm.get_buffer(target)
    except (AttributeError, TypeError):
        pass
    value: Any = gm
    try:
        for component in target.split("."):
            value = getattr(value, component)
    except AttributeError:
        return None
    return value


def _node_tensor(gm: fx.GraphModule, node: fx.Node | None) -> Any:
    if node is None:
        return None
    if node.op == "get_attr":
        value = _resolve_attr(gm, str(node.target))
        if value is not None:
            return value
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("tensor_meta")
    return value


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError, RuntimeError):
        return None
    return dims


def _linear_function(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if target is F.linear:
        return True
    return "aten.linear.default" in str(target)


def _addmm_function(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    return target is torch.addmm or "aten.addmm." in str(target)


def _addmm_weight_source(
    gm: fx.GraphModule, expression: Any
) -> tuple[fx.Node | None, tuple[int, ...] | None, str | None]:
    """Resolve a simple addmm matrix operand to its Parameter and orientation."""
    if not isinstance(expression, fx.Node):
        return None, None, None
    parameter_node = expression
    orientation: str
    if expression.op == "get_attr":
        # addmm computes mat1 @ mat2; this parameter is stored in mat2 order.
        orientation = "in_features_by_out_features"
    elif (
        expression.op == "call_method"
        and str(expression.target) == "t"
        and expression.args
        and isinstance(expression.args[0], fx.Node)
        and expression.args[0].op == "get_attr"
    ) or (
        expression.op == "call_function"
        and str(expression.target).lower().startswith("aten.t.")
        and expression.args
        and isinstance(expression.args[0], fx.Node)
        and expression.args[0].op == "get_attr"
    ):
        parameter_node = expression.args[0]
        orientation = "out_features_by_in_features"
    elif (
        expression.op == "call_method"
        and str(expression.target) == "transpose"
        and len(expression.args) >= 3
        and isinstance(expression.args[0], fx.Node)
        and expression.args[0].op == "get_attr"
        and {expression.args[1], expression.args[2]} == {0, 1}
    ):
        parameter_node = expression.args[0]
        orientation = "out_features_by_in_features"
    else:
        return None, None, None
    value = _resolve_attr(gm, str(parameter_node.target))
    shape = _shape(value)
    if shape is None or len(shape) != 2 or not isinstance(value, nn.Parameter):
        return parameter_node, shape, orientation
    return parameter_node, shape, orientation


def is_functional_linear_operation(node: fx.Node) -> bool:
    """Whether an FX function node is a visible functional linear projection."""
    return _linear_function(node)


def functional_linear_output_features(
    gm: fx.GraphModule, node: fx.Node
) -> int | None:
    """Return a recognized functional Linear's output width, if known."""
    if not _linear_function(node) or len(node.args) < 2:
        return None
    weight = node.args[1]
    if not isinstance(weight, fx.Node) or weight.op != "get_attr":
        return None
    shape = _shape(_node_tensor(gm, weight))
    return shape[0] if shape is not None and len(shape) == 2 else None


def describe_linear_operations(gm: fx.GraphModule) -> list[LinearOperationDescriptor]:
    """Describe visible module, functional Linear, and simple ``addmm`` projections.

    A functional linear node is recognized only when its weight is an explicit
    FX node backed by a Parameter. Recognition describes the projection itself;
    graph-wide resize eligibility is decided separately by
    ``FunctionalLinearResizeAnalyzer``, which checks consumers, supported
    feature-preserving paths, parameter aliases, and available tensor metadata.
    """
    aliases_by_identity: dict[int, list[str]] = {}
    try:
        for name, parameter in gm.named_parameters(remove_duplicate=False):
            aliases_by_identity.setdefault(id(parameter), []).append(name)
    except TypeError:  # Older PyTorch versions do not accept remove_duplicate.
        for name, parameter in gm.named_parameters():
            aliases_by_identity.setdefault(id(parameter), []).append(name)

    descriptors: list[LinearOperationDescriptor] = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            try:
                module = gm.get_submodule(str(node.target))
            except AttributeError:
                continue
            if not isinstance(module, nn.Linear):
                continue
            weight_node = next(
                (arg for arg in node.all_input_nodes if arg.op == "get_attr" and str(arg.target).endswith("weight")),
                None,
            )
            bias_node = next(
                (arg for arg in node.all_input_nodes if arg.op == "get_attr" and str(arg.target).endswith("bias")),
                None,
            )
            aliases = tuple(aliases_by_identity.get(id(module.weight), ()))
            descriptors.append(LinearOperationDescriptor(
                node=node,
                form="module",
                input_node=next(iter(node.all_input_nodes), None),
                weight_node=weight_node,
                bias_node=bias_node,
                weight_target=(str(weight_node.target) if weight_node is not None else f"{node.target}.weight"),
                parameter_aliases=aliases,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                weight_orientation="out_features_by_in_features",
                recognized=True,
                resize_support="module_linear_rule",
            ))
            continue

        if _addmm_function(node):
            input_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
            bias_node = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
            weight_node, weight_shape, orientation = _addmm_weight_source(
                gm, node.args[2] if len(node.args) > 2 else None
            )
            weight_target = (
                str(weight_node.target)
                if weight_node is not None and weight_node.op == "get_attr"
                else None
            )
            weight_parameter = _resolve_attr(gm, weight_target) if weight_target is not None else None
            aliases = (
                tuple(aliases_by_identity.get(id(weight_parameter), ()))
                if isinstance(weight_parameter, nn.Parameter)
                else ()
            )
            reasons: list[str] = []
            if input_node is None:
                reasons.append("addmm projection input is not an explicit FX node")
            if weight_node is None:
                reasons.append("addmm matrix operand is not a supported Parameter or transpose thereof")
            elif weight_shape is None or len(weight_shape) != 2:
                reasons.append("addmm weight shape is missing or is not rank 2")
            elif not isinstance(weight_parameter, nn.Parameter):
                reasons.append("addmm weight target does not resolve to a Parameter")
            in_features = out_features = None
            if weight_shape is not None and len(weight_shape) == 2 and orientation is not None:
                if orientation == "out_features_by_in_features":
                    out_features, in_features = weight_shape
                else:
                    in_features, out_features = weight_shape
            descriptors.append(LinearOperationDescriptor(
                node=node,
                form="addmm",
                input_node=input_node,
                weight_node=weight_node,
                bias_node=bias_node,
                weight_target=weight_target,
                parameter_aliases=aliases,
                in_features=in_features,
                out_features=out_features,
                weight_orientation=orientation or "unknown",
                recognized=not reasons,
                resize_support="requires_chain_analysis",
                diagnostic=(
                    "; ".join(reasons)
                    if reasons
                    else "addmm projection is recognized; resize safety depends on its graph path and parameter aliases"
                ),
            ))
            continue

        if not _linear_function(node):
            continue
        input_node = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        weight_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        bias_node = node.args[2] if len(node.args) > 2 and isinstance(node.args[2], fx.Node) else None
        weight_target = str(weight_node.target) if weight_node is not None and weight_node.op == "get_attr" else None
        weight_value = _node_tensor(gm, weight_node)
        weight_shape = _shape(weight_value)
        weight_parameter = _resolve_attr(gm, weight_target) if weight_target is not None else None
        aliases = tuple(aliases_by_identity.get(id(weight_parameter), ())) if isinstance(weight_parameter, nn.Parameter) else ()
        reasons: list[str] = []
        if input_node is None:
            reasons.append("functional linear input is not an explicit FX node")
        if weight_node is None or weight_node.op != "get_attr":
            reasons.append("functional linear weight is not a graph get_attr parameter")
        if weight_shape is None or len(weight_shape) != 2:
            reasons.append("functional linear weight shape is missing or is not rank 2")
        elif weight_target is None:
            reasons.append("functional linear weight target cannot be resolved")
        if weight_target is not None and not isinstance(weight_parameter, nn.Parameter):
            reasons.append("functional linear weight target does not resolve to a Parameter")
        descriptors.append(LinearOperationDescriptor(
            node=node,
            form="function",
            input_node=input_node,
            weight_node=weight_node,
            bias_node=bias_node,
            weight_target=weight_target,
            parameter_aliases=aliases,
            in_features=weight_shape[1] if weight_shape is not None and len(weight_shape) == 2 else None,
            out_features=weight_shape[0] if weight_shape is not None and len(weight_shape) == 2 else None,
            weight_orientation="out_features_by_in_features",
            recognized=not reasons,
            resize_support="requires_chain_analysis",
            diagnostic=(
                "; ".join(reasons)
                if reasons
                else "functional projection is recognized; resize safety depends on its graph path and parameter aliases"
            ),
        ))
    return descriptors
