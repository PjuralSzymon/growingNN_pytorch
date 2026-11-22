"""Per-node queries: module resolution, type checks, and feature-width analysis."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from growingnn.utils.fx.sum_nodes import is_sum_node

from growingnn.core.config import (
    PASSTHROUGH_MODULES,
    PASSTHROUGH_MODULES_TO_UPDATE,
    PASSTHROUGH_FUNCTIONS,
    PROPAGATION_RESIZABLE_MODULES,
)


# Operations that move or merge the feature dimension, so a new width changes what they compute.
RESHAPING_OPERATIONS = frozenset({torch.cat, torch.concat, torch.reshape, torch.permute, torch.transpose,
                                  "view", "reshape", "permute", "transpose"})


class PropagationVerdict(Enum):
    """Whether a new layer width can be propagated through the rest of the graph."""

    SAFE = "safe"
    UNKNOWN = "unknown"  # the width reaches an operation the rules do not know
    BLOCKED = "blocked"


class _WidthStep(Enum):
    """How an operation treats a new width on its input."""

    ABSORBS = "absorbs"  # reads it with resizable weights; its own output keeps its width
    CARRIES = "carries"  # passes it on unchanged
    BLOCKS = "blocks"
    UNKNOWN = "unknown"


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
    def is_global_pool(node: fx.Node, gm: fx.GraphModule) -> bool:
        """True for adaptive 2-D pooling to a 1x1 output."""
        if NodeTypeChecker.two_d_pool_kind(node, gm) not in ("adaptive_max", "adaptive_avg"):
            return False
        if node.op == "call_module":
            size = ModuleResolver.get_layer_module(node, gm).output_size
        else:
            size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size")
        return size in (1, (1, 1))

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
        """Output channel width reading live module attributes, walking through passthroughs and adds."""
        if n.op == "call_module":
            m = ModuleResolver.get_layer_module(n.target, gm)
            if isinstance(m, nn.Linear):
                return m.out_features
            if isinstance(m, nn.Conv2d):
                return m.out_channels
            if isinstance(m, PASSTHROUGH_MODULES_TO_UPDATE):
                return m.num_features
            if isinstance(m, nn.LayerNorm):
                shape = m.normalized_shape
                return int(shape[-1]) if isinstance(shape, tuple) and shape else int(shape)
            if isinstance(m, nn.Sequential):
                # Residual Sequential (e.g. zero-conv + pool + flatten): use last Conv/Linear width.
                width = None
                for child in m.modules():
                    if isinstance(child, nn.Conv2d):
                        width = child.out_channels
                    elif isinstance(child, nn.Linear):
                        width = child.out_features
                return width
        # Flatten keeps the width only when nothing spatial is left: after global pooling, or on
        # (batch, features) input. After other pooling its width is channels x height x width.
        if NodeTypeChecker.is_flatten_node(n, gm) and n.all_input_nodes:
            source = n.all_input_nodes[0]
            if NodeTypeChecker.is_global_pool(source, gm):
                return NodeWidthAnalyser.node_output_width(gm, source)
            if source.op == "call_module" and isinstance(ModuleResolver.get_layer_module(source.target, gm), nn.Linear):
                return NodeWidthAnalyser.node_output_width(gm, source)
            return None
        # 2-D pooling keeps the channel count.
        if (NodeTypeChecker.is_passthrough(gm, n) or NodeTypeChecker.is_add(n)
                or NodeTypeChecker.is_pool_node(n, gm)) and n.all_input_nodes:
            return NodeWidthAnalyser.node_output_width(gm, n.all_input_nodes[0])
        return None

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
    def _width_step(gm: fx.GraphModule, node: fx.Node) -> _WidthStep:
        """How *node* treats a new width on its input."""
        if node.op == "output":
            return _WidthStep.ABSORBS
        if NodeTypeChecker.is_flatten_node(node, gm):
            source = node.all_input_nodes[0]
            keeps_width = NodeTypeChecker.is_global_pool(source, gm) or (
                source.op == "call_module"
                and isinstance(ModuleResolver.get_layer_module(source.target, gm), nn.Linear))
            return _WidthStep.CARRIES if keeps_width else _WidthStep.BLOCKS
        if node.op == "call_module":
            mod = ModuleResolver.get_layer_module(node.target, gm)
            if isinstance(mod, PROPAGATION_RESIZABLE_MODULES):
                return _WidthStep.ABSORBS
            if isinstance(mod, nn.Sequential):
                return _WidthStep.ABSORBS if NodeWidthAnalyser._sequential_branch_resizable(mod) else _WidthStep.BLOCKS
            if isinstance(mod, (PASSTHROUGH_MODULES, PASSTHROUGH_MODULES_TO_UPDATE)):
                return _WidthStep.CARRIES
            return _WidthStep.BLOCKS
        if (NodeTypeChecker.is_add(node) or NodeTypeChecker.is_pool_node(node, gm)
                or NodeTypeChecker.is_passthrough(gm, node)):
            return _WidthStep.CARRIES
        if node.target in RESHAPING_OPERATIONS:
            return _WidthStep.BLOCKS
        return _WidthStep.UNKNOWN

    @staticmethod
    def _queue_resized_partner(gm: fx.GraphModule, partner: fx.Node, pending: list[fx.Node]) -> PropagationVerdict:
        """Walk up from a sum input to the layers that must be resized to match it, queueing every node
        whose width changes with them."""
        stack, seen = [partner], set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.op == "placeholder":
                return PropagationVerdict.BLOCKED
            step = NodeWidthAnalyser._width_step(gm, node)
            if step is _WidthStep.BLOCKS:
                return PropagationVerdict.BLOCKED
            if step is _WidthStep.UNKNOWN:
                return PropagationVerdict.UNKNOWN
            pending.append(node)
            if step is _WidthStep.CARRIES:
                stack.extend(node.all_input_nodes)
        return PropagationVerdict.SAFE

    @staticmethod
    def propagation_verdict(gm: fx.GraphModule, start_node: fx.Node) -> PropagationVerdict:
        """Whether a new output width of *start_node* can be propagated through the graph.

        The width travels through operations that keep it (activations, BatchNorm, pooling, sums) and
        stops at the convolutions and linear layers that read it. A sum also resizes its other inputs,
        whose other readers are checked the same way.
        """
        verdict = PropagationVerdict.SAFE
        pending, seen = [start_node], set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            for user in node.users:
                step = NodeWidthAnalyser._width_step(gm, user)
                if step is _WidthStep.BLOCKS:
                    return PropagationVerdict.BLOCKED
                if step is _WidthStep.UNKNOWN:
                    verdict = PropagationVerdict.UNKNOWN
                    continue
                if step is _WidthStep.ABSORBS:
                    continue
                if NodeTypeChecker.is_add(user):
                    for partner in user.all_input_nodes:
                        if partner is node:
                            continue
                        partner_verdict = NodeWidthAnalyser._queue_resized_partner(gm, partner, pending)
                        if partner_verdict is PropagationVerdict.BLOCKED:
                            return partner_verdict
                        if partner_verdict is PropagationVerdict.UNKNOWN:
                            verdict = partner_verdict
                pending.append(user)
        return verdict

    @staticmethod
    def propagation_hits_unsizable(gm: fx.GraphModule, start_node: fx.Node) -> bool:
        """True unless the rules show that a new width of *start_node* propagates safely."""
        return NodeWidthAnalyser.propagation_verdict(gm, start_node) is not PropagationVerdict.SAFE
