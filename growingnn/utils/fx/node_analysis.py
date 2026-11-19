"""Per-node queries: module resolution, type checks, and feature-width analysis."""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from growingnn.utils.fx.sum_nodes import is_sum_node
from growingnn.utils.fx.operation_descriptors import (
    functional_linear_output_features,
    is_functional_linear_operation,
)

from growingnn.core.config import (
    PASSTHROUGH_MODULES,
    PASSTHROUGH_MODULES_TO_UPDATE,
    PASSTHROUGH_FUNCTIONS,
    PROPAGATION_RESIZABLE_MODULES,
)


class ModuleResolver:
    """Resolve, find, and generate names for submodules inside an FX GraphModule."""

    @staticmethod
    def get_layer_module(target: fx.Node | str, gm: nn.Module | fx.GraphModule) -> nn.Module | None:
        """Resolve a submodule by FX node or dotted path. Returns None when the path does not exist."""
        name = target.target if isinstance(target, fx.Node) else target
        try:
            return gm.get_submodule(str(name))
        except AttributeError:
            return None

    @staticmethod
    def find_call_module(nodes, target_name: str) -> fx.Node:
        """Return the first call_module node whose target matches *target_name*."""
        for n in nodes:
            if n.op == "call_module" and n.target == target_name:
                return n
        available = sorted({n.target for n in nodes if n.op == "call_module"})
        raise ValueError(
            f"No call_module node with target {target_name!r}. "
            f"Available call_module targets: {available}"
        )

    @staticmethod
    def unique_call_module_name(base: str, model: nn.Module | fx.GraphModule) -> str:
        """Pick a non-colliding call_module target name derived from *base*."""
        names: set[str] = set(model._modules.keys())
        if isinstance(model, fx.GraphModule):
            names |= {str(n.target) for n in model.graph.nodes if n.op == "call_module"}

        suffixes: list[int] = []
        for n in names:
            if n == base:
                suffixes.append(0)
            elif n.startswith(base + "_"):
                rest = n[len(base) + 1:]
                if rest.isdigit():
                    suffixes.append(int(rest))

        if not suffixes:
            return base + "_0"
        return base + "_" + str(max(suffixes) + 1)


class NodeTypeChecker:
    """Boolean queries about a single FX node's role in the graph."""

    @staticmethod
    def is_passthrough(gm: fx.GraphModule, n: fx.Node) -> bool:
        """True for nodes that forward tensors without changing their shape."""
        return (n.op == "call_function" and n.target in PASSTHROUGH_FUNCTIONS) or \
               (n.op == "call_module" and isinstance(ModuleResolver.get_layer_module(n.target, gm), PASSTHROUGH_MODULES)) or \
                (n.op == "call_method" and n.target in PASSTHROUGH_FUNCTIONS)

    @staticmethod
    def is_fork(n: fx.Node) -> bool:
        """True when the node has more than one user."""
        return len(n.users) > 1

    @staticmethod
    def is_add(n: fx.Node) -> bool:
        """True for binary or variadic tensor-sum call_function nodes."""
        return is_sum_node(n)

    @staticmethod
    def is_elementwise(n: fx.Node) -> bool:
        """True for common pointwise arithmetic operations in FX graphs."""
        if n.op not in ("call_function", "call_method"):
            return False
        if n.target in {
            operator.add, operator.sub, operator.mul, operator.truediv,
            operator.floordiv, operator.pow, torch.add, torch.sub, torch.mul,
            torch.div, torch.pow, torch.tanh,
        }:
            return True
        if n.op == "call_method" and str(n.target).lower() in {
            "add", "sub", "mul", "div", "pow", "tanh"
        }:
            return True
        # Export graphs use overload objects such as aten.mul.Tensor. Comparing
        # their stable namespace spelling keeps this independent of PyTorch's
        # generated overload object identities.
        target_name = str(n.target).lower()
        return any(
            target_name.startswith("aten." + operation + ".")
            for operation in ("add", "sub", "mul", "div", "pow", "tanh")
        )

    @staticmethod
    def is_feature_layout_change(n: fx.Node) -> bool:
        """True for operations that partition or rearrange tensor dimensions."""
        if n.op not in ("call_function", "call_method"):
            return False
        target_name = str(n.target).lower()
        return any(
            token in target_name
            for token in (
                "split", "chunk", "unbind", "view", "reshape", "flatten",
                "permute", "transpose", "movedim", "swapaxes", "select", "narrow",
            )
        )

    @staticmethod
    def is_flatten_node(node: fx.Node, gm: fx.GraphModule) -> bool:
        """True when the node is a method, module, or function flatten site."""
        if node.op == "call_method" and node.target == "flatten":
            return True
        if node.op == "call_function" and node.target is torch.flatten:
            return True
        if node.op == "call_module":
            module = ModuleResolver.get_layer_module(node, gm)
            return isinstance(module, nn.Flatten)
        return False

    @staticmethod
    def is_pool_node(node: fx.Node, gm: fx.GraphModule) -> bool:
        """True when the node is a functional or module 2-D pooling site."""
        return NodeTypeChecker.two_d_pool_kind(node, gm) is not None

    @staticmethod
    def two_d_pool_kind(node: fx.Node, gm: fx.GraphModule) -> str | None:
        """Return windowed_max/windowed_avg/adaptive_max/adaptive_avg, else None."""
        if node.op == "call_function":
            if node.target is F.max_pool2d:
                return "windowed_max"
            if node.target is F.avg_pool2d:
                return "windowed_avg"
            if node.target is F.adaptive_max_pool2d:
                return "adaptive_max"
            if node.target is F.adaptive_avg_pool2d:
                return "adaptive_avg"
            return None
        if node.op != "call_module":
            return None
        module = ModuleResolver.get_layer_module(node, gm)
        if isinstance(module, nn.MaxPool2d):
            return "windowed_max"
        if isinstance(module, nn.AvgPool2d):
            return "windowed_avg"
        if isinstance(module, nn.AdaptiveMaxPool2d):
            return "adaptive_max"
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return "adaptive_avg"
        return None


class NodeWidthAnalyser:
    """Feature-width queries and propagation safety checks on FX graph nodes."""

    @staticmethod
    def node_output_width(gm: fx.GraphModule, n: fx.Node) -> int | None:
        """Read live module widths and propagate them through known pointwise FX ops.

        This is iterative so a long functional chain does not consume Python
        recursion depth. Unknown shape-changing operations deliberately return
        ``None`` rather than guessing a feature dimension.
        """
        widths: dict[fx.Node, int | None] = {}
        stack: list[tuple[fx.Node, bool]] = [(n, False)]

        def module_width(node: fx.Node) -> int | None:
            functional_width = functional_linear_output_features(gm, node)
            if functional_width is not None:
                return functional_width
            if node.op != "call_module":
                return None
            module = ModuleResolver.get_layer_module(node.target, gm)
            if isinstance(module, nn.Linear):
                return int(module.out_features)
            if isinstance(module, nn.Conv2d):
                return int(module.out_channels)
            if isinstance(module, PASSTHROUGH_MODULES_TO_UPDATE):
                return int(module.num_features)
            if isinstance(module, nn.LayerNorm):
                shape = module.normalized_shape
                return int(shape[-1]) if isinstance(shape, tuple) and shape else int(shape)
            if isinstance(module, nn.Sequential):
                # Retain the legacy residual-Sequential width convention.
                result = None
                for child in module.modules():
                    if isinstance(child, nn.Conv2d):
                        result = int(child.out_channels)
                    elif isinstance(child, nn.Linear):
                        result = int(child.out_features)
                return result
            return None

        def scalar_value(node: fx.Node) -> bool:
            value = node.meta.get("val")
            if isinstance(value, (bool, int, float, complex)):
                return True
            metadata = node.meta.get("tensor_meta")
            if metadata is None and value is not None:
                metadata = value
            shape = getattr(metadata, "shape", None)
            return shape is not None and len(shape) == 0

        def dependencies(node: fx.Node) -> list[fx.Node] | None:
            if NodeTypeChecker.is_passthrough(gm, node):
                return list(node.all_input_nodes)
            if NodeTypeChecker.is_add(node) or NodeTypeChecker.is_elementwise(node):
                return [dep for dep in node.all_input_nodes if not scalar_value(dep)]
            return None

        while stack:
            node, expanded = stack.pop()
            if node in widths:
                continue
            direct = module_width(node)
            if direct is not None:
                widths[node] = direct
                continue
            deps = dependencies(node)
            if deps is None or not deps:
                widths[node] = None
                continue
            if not expanded:
                stack.append((node, True))
                stack.extend((dep, False) for dep in deps if dep not in widths)
                continue
            dep_widths = [widths.get(dep) for dep in deps]
            widths[node] = (
                dep_widths[0]
                if dep_widths[0] is not None and all(width == dep_widths[0] for width in dep_widths)
                else None
            )
        return widths.get(n)

    @staticmethod
    def inputs_match_width(gm: fx.GraphModule, n: fx.Node, w: int) -> bool:
        """True when every input to node n has output width w."""
        return bool(n.all_input_nodes) and all(
            NodeWidthAnalyser.node_output_width(gm, i) == w for i in n.all_input_nodes
        )

    @staticmethod
    def all_sites_match_width(gm: fx.GraphModule, module_name: str, w: int) -> bool:
        """True when every call_module site for module_name has all inputs at width w."""
        return all(NodeWidthAnalyser.inputs_match_width(gm, n, w)
                   for n in gm.graph.nodes if n.op == "call_module" and n.target == module_name)

    @staticmethod
    def _sequential_branch_resizable(mod: nn.Module) -> bool:
        if isinstance(mod, PROPAGATION_RESIZABLE_MODULES) or isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
            return True
        if isinstance(mod, PASSTHROUGH_MODULES):
            return False
        if isinstance(mod, nn.Sequential):
            return any(NodeWidthAnalyser._sequential_branch_resizable(c) for c in mod.children())
        return False

    @staticmethod
    def branch_has_unsizable_module(gm: fx.GraphModule, node: fx.Node, seen: set[str] | None = None) -> bool:
        """True when walking backward from node hits a module propagation cannot resize."""
        local_seen = seen if seen is not None else set()

        def _walk(n: fx.Node) -> bool:
            if n.name in local_seen:
                return False
            local_seen.add(n.name)
            if is_functional_linear_operation(n):
                return True
            if NodeTypeChecker.is_passthrough(gm, n) or (
                n.op == "call_module"
                and isinstance(ModuleResolver.get_layer_module(n.target, gm), PASSTHROUGH_MODULES_TO_UPDATE)
            ):
                return any(_walk(inp) for inp in n.all_input_nodes)
            if n.op == "call_module":
                mod = ModuleResolver.get_layer_module(n.target, gm)
                if isinstance(mod, nn.Sequential):
                    return not NodeWidthAnalyser._sequential_branch_resizable(mod)
                return mod is not None and not isinstance(mod, PROPAGATION_RESIZABLE_MODULES)
            if n.op == "placeholder":
                return False
            return any(_walk(inp) for inp in n.all_input_nodes)

        return _walk(node)

    @staticmethod
    def propagation_hits_unsizable(gm: fx.GraphModule, start_node: fx.Node) -> bool:
        """True when width propagation encounters an unsupported boundary."""
        return NodeWidthAnalyser.propagation_resize_rejection_reason(gm, start_node) is not None

    @staticmethod
    def propagation_resize_rejection_reason(
        gm: fx.GraphModule, start_node: fx.Node
    ) -> str | None:
        """Explain the first unsafe downstream boundary for a width change."""
        seen: set[str] = set()

        def _module_blocks_resize(mod: nn.Module | None) -> bool:
            if mod is None:
                return False
            if isinstance(mod, PROPAGATION_RESIZABLE_MODULES):
                return False
            if isinstance(mod, PASSTHROUGH_MODULES) or isinstance(mod, PASSTHROUGH_MODULES_TO_UPDATE):
                return False
            if isinstance(mod, nn.Sequential):
                return not NodeWidthAnalyser._sequential_branch_resizable(mod)
            # LayerNorm and other fixed-shape modules block safe neuron resize.
            return True

        def _walk(node: fx.Node) -> str | None:
            if node.name in seen:
                return None
            seen.add(node.name)
            for user in node.users:
                if user.op == "output":
                    continue
                if is_functional_linear_operation(user):
                    return (
                        f"downstream functional projection {user.name} ({user.target!s}) is recognized, "
                        "but generic width propagation does not apply a validated functional-chain resize"
                    )
                if NodeTypeChecker.is_add(user):
                    for inp in user.all_input_nodes:
                        if inp is not node and NodeWidthAnalyser.branch_has_unsizable_module(gm, inp, seen):
                            return (
                                f"merge {user.name} has a sibling branch that cannot be resized "
                                f"with the output of {node.name}"
                            )
                if user.op == "call_module":
                    mod = ModuleResolver.get_layer_module(user.target, gm)
                    if _module_blocks_resize(mod):
                        return (
                            f"downstream module {user.target!s} ({type(mod).__name__ if mod is not None else 'unknown'}) "
                            "has no supported width-resize rule"
                        )
                    if isinstance(mod, (nn.Linear, nn.Conv2d)):
                        # These projections can consume the new input width
                        # while keeping their own output width fixed. The width
                        # fix pass updates their input parameters, so downstream
                        # residual branches remain at their original width.
                        continue
                if NodeTypeChecker.is_feature_layout_change(user):
                    # A width change can alter partition counts, head layouts,
                    # or fixed reshape dimensions. Without a transform-specific
                    # contract the downstream interpretation is not preserved.
                    return (
                        f"downstream operation {user.name} ({user.target!s}) changes or partitions "
                        "feature layout; coordinated dimension changes are unsupported"
                    )
                reason = _walk(user)
                if reason is not None:
                    return reason
            return None

        return _walk(start_node)
