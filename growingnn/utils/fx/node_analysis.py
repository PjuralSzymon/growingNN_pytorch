"""Per-node queries: module resolution, type checks, and feature-width analysis."""

from __future__ import annotations

import torch.nn as nn
import torch.fx as fx

from growingnn.utils.fx.sum_nodes import is_sum_node

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


class NodeWidthAnalyser:
    """Feature-width queries and propagation safety checks on FX graph nodes."""

    @staticmethod
    def node_output_width(gm: fx.GraphModule, n: fx.Node) -> int | None:
        """Output channel width reading live module attributes, walking through passthroughs and adds."""
        if n.op == "call_module":
            m = ModuleResolver.get_layer_module(n.target, gm)
            if isinstance(m, nn.Linear): return m.out_features
            if isinstance(m, nn.Conv2d): return m.out_channels
            if isinstance(m, nn.LayerNorm): return m.normalized_shape[0]
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): return m.num_features
        if (NodeTypeChecker.is_passthrough(gm, n) or NodeTypeChecker.is_add(n)) and n.all_input_nodes:
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
    def branch_has_unsizable_module(gm: fx.GraphModule, node: fx.Node, seen: set[str] | None = None) -> bool:
        """True when walking backward from node hits a module propagation cannot resize."""
        local_seen = seen if seen is not None else set()

        def _walk(n: fx.Node) -> bool:
            if n.name in local_seen:
                return False
            local_seen.add(n.name)
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
        """True if forward propagation would hit an add whose sibling branch cannot be width-synced."""
        seen: set[str] = set()

        def _walk(node: fx.Node) -> bool:
            if node.name in seen:
                return False
            seen.add(node.name)
            for user in node.users:
                if user.op == "output":
                    continue
                if NodeTypeChecker.is_add(user):
                    for inp in user.all_input_nodes:
                        if inp is not node and NodeWidthAnalyser.branch_has_unsizable_module(gm, inp, seen):
                            return True
                if _walk(user):
                    return True
            return False

        return _walk(start_node)
