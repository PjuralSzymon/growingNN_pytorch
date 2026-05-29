"""Per-node queries: module resolution, type checks, and feature-width analysis."""

from __future__ import annotations

import operator

import torch.nn as nn
import torch.fx as fx

from growingnn.core.config import PASSTHROUGH_MODULES, PASSTHROUGH_MODULES_TO_UPDATE, PASSTHROUGH_FUNCTIONS, RESIZE_SAFE_MODULES


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
        """True for operator.add call_function nodes."""
        return n.op == "call_function" and n.target == operator.add


class NodeWidthAnalyser:
    """Feature-width queries and propagation safety checks on FX graph nodes."""

    @staticmethod
    def node_output_width(gm: fx.GraphModule, n: fx.Node) -> int | None:
        """Output channel width reading live module attributes, walking through passthroughs and adds."""
        if n.op == "call_module":
            m = ModuleResolver.get_layer_module(n.target, gm)
            if isinstance(m, nn.Linear): return m.out_features
            if isinstance(m, nn.Conv2d): return m.out_channels
            if isinstance(m, PASSTHROUGH_MODULES_TO_UPDATE): return m.num_features
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
    def propagation_hits_unsizable(gm: fx.GraphModule, start_node: fx.Node) -> bool:
        """True if forward propagation from start_node would reach an add whose
        sibling branch contains a non-resizable call_module (e.g. Conv2d)."""
        seen: set[str] = set()

        def _check_sibling(node: fx.Node) -> bool:
            if node.name in seen:
                return False
            seen.add(node.name)
            if NodeTypeChecker.is_passthrough(gm, node) or (
                node.op == "call_module" and isinstance(ModuleResolver.get_layer_module(node.target, gm), PASSTHROUGH_MODULES_TO_UPDATE)
            ):
                return any(_check_sibling(inp) for inp in node.all_input_nodes)
            if node.op == "call_module":
                mod = ModuleResolver.get_layer_module(node.target, gm)
                if mod is not None and not isinstance(mod, RESIZE_SAFE_MODULES):
                    return True
                return False
            if node.op == "placeholder":
                return False
            return any(_check_sibling(inp) for inp in node.all_input_nodes)

        def _walk(node: fx.Node) -> bool:
            if node.name in seen:
                return False
            seen.add(node.name)
            for user in node.users:
                if user.op == "output":
                    continue
                if NodeTypeChecker.is_add(user):
                    for inp in user.all_input_nodes:
                        if inp is not node and _check_sibling(inp):
                            return True
                if _walk(user):
                    return True
            return False

        return _walk(start_node)
