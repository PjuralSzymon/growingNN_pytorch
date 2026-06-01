"""Whole-graph read-only queries: module classification, topology, shape propagation, bridge sizing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from growingnn.core.config import EDITABLE_MODULES
from growingnn.core.logger import logger
from growingnn.utils.fx.node_analysis import ModuleResolver


class ModuleClassifier:
    """Classify call_module nodes by their position and role in the graph."""

    @staticmethod
    def is_hidden_module(node: fx.Node) -> bool:
        """True for call_module nodes that sit strictly between the input and output boundaries."""
        if len(node.users) == 0 or len(node.all_input_nodes) == 0:
            return False
        if any(user.op == "output" for user in node.users):
            return False
        if "placeholder" in node.all_input_nodes:
            return False
        if len(node.all_input_nodes) == 1:
            inp = node.all_input_nodes[0]
            if inp is None:
                return False
            if len(inp.all_input_nodes) == 0:
                return False
        return True

    @staticmethod
    def is_editable_module(node: fx.Node, gm: fx.GraphModule) -> bool:
        """True when the node is a call_module whose resolved type is in EDITABLE_MODULES."""
        if node.op != "call_module":
            return False
        module = ModuleResolver.get_layer_module(node, gm)
        if module is None:
            return False
        return any(isinstance(module, t) for t in EDITABLE_MODULES)

    @staticmethod
    def is_at_least_one_hidden_module(n1: fx.Node, n2: fx.Node) -> bool:
        """True when at least one of the two nodes is a hidden module."""
        return ModuleClassifier.is_hidden_module(n1) or ModuleClassifier.is_hidden_module(n2)

    @staticmethod
    def is_edge_into_hidden_module(src: fx.Node, dst: fx.Node) -> bool:
        """True for visible->hidden and hidden->hidden edges; false otherwise."""
        return ModuleClassifier.is_hidden_module(dst)


class GraphStructureQuery:
    """Structural queries over the full FX graph: hidden modules, pairs, adjacency, parameters."""

    @staticmethod
    def get_all_hidden_modules(model: nn.Module | fx.GraphModule) -> list[str]:
        """Return the target names of all hidden call_module nodes."""
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        nodes: list[str] = []
        for n in gm.graph.nodes:
            if n.op != "call_module":
                continue
            if not ModuleClassifier.is_hidden_module(n):
                logger.debug("n.target: %s is not a hidden module", n.target)
                continue
            nodes.append(str(n.target))
        return nodes

    @staticmethod
    def module_dependency_pairs(model: nn.Module | fx.GraphModule) -> list[tuple[str, str]]:
        """All (ancestor, descendant) pairs where the descendant is a hidden module reachable forward."""
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        edges: list[tuple[str, str]] = []
        for n in gm.graph.nodes:
            if not ModuleClassifier.is_editable_module(n, gm):
                continue
            src = str(n.target)
            stack, seen = list(n.users), set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if ModuleClassifier.is_editable_module(cur, gm) and ModuleClassifier.is_edge_into_hidden_module(n, cur):
                    edges.append((src, str(cur.target)))
                stack.extend(cur.users)
        logger.debug("number of dependency pairs: %s", len(edges))
        return list(dict.fromkeys(edges))

    @staticmethod
    def module_sequential_pairs(model: nn.Module | fx.GraphModule) -> list[tuple[str, str]]:
        """All (ancestor, descendant) pairs that are next to each other in the model."""
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        edges: list[tuple[str, str]] = []
        for n in gm.graph.nodes:
            if not ModuleClassifier.is_editable_module(n, gm):
                continue
            src = str(n.target)
            stack, seen = list(n.users), set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if ModuleClassifier.is_editable_module(cur, gm) and ModuleClassifier.is_at_least_one_hidden_module(n, cur):
                    edges.append((src, str(cur.target)))
                    continue
                stack.extend(cur.users)
        logger.debug("number of sequential pairs: %s", len(edges))
        return list(dict.fromkeys(edges))

    @staticmethod
    def _sum_graph_module_parameters(gm: fx.GraphModule) -> int:
        """Sum unique parameters reachable from call_module nodes in the FX graph."""
        seen: set[int] = set()
        total = 0
        for node in gm.graph.nodes:
            if node.op != "call_module":
                continue
            module = ModuleResolver.get_layer_module(node, gm)
            if module is None:
                continue
            for param in module.parameters(recurse=True):
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                total += param.numel()
        return total

    @staticmethod
    def get_amount_of_parameters(model: nn.Module | fx.GraphModule) -> int:
        """Total number of parameters in FX graph call_module nodes (conv, linear, etc.)."""
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        return GraphStructureQuery._sum_graph_module_parameters(gm)

    @staticmethod
    def get_input_layers(layer_id: str, model: nn.Module | fx.GraphModule) -> list[str]:
        """Predecessor editable modules of *layer_id*."""
        pred, _ = GraphStructureQuery._sequential_adj(model)
        return list(pred.get(layer_id, []))

    @staticmethod
    def get_output_layers(layer_id: str, model: nn.Module | fx.GraphModule) -> list[str]:
        """Successor editable modules of *layer_id*."""
        _, succ = GraphStructureQuery._sequential_adj(model)
        return list(succ.get(layer_id, []))

    @staticmethod
    def _sequential_adj(model: nn.Module | fx.GraphModule) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Build predecessor and successor adjacency dicts from sequential pairs."""
        pred: dict[str, list[str]] = {}
        succ: dict[str, list[str]] = {}
        for a, b in dict.fromkeys(GraphStructureQuery.module_sequential_pairs(model)):
            pred.setdefault(b, []).append(a)
            succ.setdefault(a, []).append(b)
        return pred, succ


class LayerShapeAnalyser:
    """Run ShapeProp and read per-layer input/output activation shapes."""

    @staticmethod
    def node_shape(node: fx.Node) -> tuple[int, ...] | None:
        """Extract the output shape from a node's metadata."""
        meta = node.meta
        val = meta.get("val")
        if val is not None and hasattr(val, "shape"):
            return tuple(int(x) for x in val.shape)
        tm = meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape"):
            return tuple(int(x) for x in tm.shape)
        return None

    @staticmethod
    def default_example_input(gm: fx.GraphModule) -> torch.Tensor | None:
        """Guess a suitable example input tensor from the first layer found."""
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if not placeholders:
            return None
        try:
            p0 = next(gm.parameters())
            device, dtype = p0.device, p0.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32
        for mod in gm.modules():
            if isinstance(mod, nn.Linear):
                return torch.randn(1, mod.in_features, device=device, dtype=dtype)
            if isinstance(mod, nn.modules.conv._ConvNd):
                return torch.randn(1, mod.in_channels, 224, 224, device=device, dtype=dtype)
        return torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

    @staticmethod
    def input_shape_for_layer(node: fx.Node) -> tuple[int, ...] | None:
        """Read the activation shape entering a layer node from its first argument."""
        if not node.args:
            return None
        arg0 = node.args[0]
        if isinstance(arg0, fx.Node):
            return LayerShapeAnalyser.node_shape(arg0)
        return None

    @staticmethod
    def collect_layer_shapes(
        gm: fx.GraphModule,
        example: torch.Tensor | None = None,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        """Run ShapeProp and return (output_shapes, input_shapes) dicts keyed by module target."""
        outputs: dict[str, tuple[int, ...]] = {}
        inputs: dict[str, tuple[int, ...]] = {}
        probe = example if example is not None else LayerShapeAnalyser.default_example_input(gm)
        if probe is None:
            return outputs, inputs
        try:
            ShapeProp(gm).propagate(probe)
        except Exception:
            return outputs, inputs
        for node in gm.graph.nodes:
            if node.op != "call_module" or not isinstance(node.target, str):
                continue
            out_shape = LayerShapeAnalyser.node_shape(node)
            if out_shape is not None:
                outputs[node.target] = out_shape
            in_shape = LayerShapeAnalyser.input_shape_for_layer(node)
            if in_shape is not None:
                inputs[node.target] = in_shape
        return outputs, inputs

    @staticmethod
    def get_layer_output_shapes(
        gm: fx.GraphModule,
        example: torch.Tensor | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Output activation shapes for every call_module node."""
        return LayerShapeAnalyser.collect_layer_shapes(gm, example)[0]

    @staticmethod
    def get_layer_input_shapes(
        gm: fx.GraphModule,
        example: torch.Tensor | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Input activation shapes for every call_module node."""
        return LayerShapeAnalyser.collect_layer_shapes(gm, example)[1]


class LayerBridgeFinder:
    """From activation shapes, decide if a bridge layer fits and what sizes it needs."""

    @staticmethod
    def uniform_activation_shape(
        shapes: list[tuple[int, ...] | None],
    ) -> tuple[int, ...] | None:
        """Return the common shape if all entries are identical, else None."""
        if not shapes or any(s is None for s in shapes):
            return None
        first = shapes[0]
        return first if all(s == first for s in shapes) else None

    @staticmethod
    def linear_feature_dim(shape: tuple[int, ...]) -> int | None:
        """Feature dimension from a 2-D (batch, features) shape."""
        if len(shape) != 2:
            return None
        features = int(shape[1])
        return features if features > 0 else None

    @staticmethod
    def conv_channels(shape: tuple[int, ...]) -> int | None:
        """Channel dimension from a 4-D (batch, channels, H, W) shape."""
        if len(shape) != 4:
            return None
        channels = int(shape[1])
        return channels if channels > 0 else None

    @staticmethod
    def find_bridge_linear_sizes(
        from_output_shape: tuple[int, ...] | None,
        to_input_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """(in_features, out_features) for a bridge Linear between two layers."""
        if from_output_shape is None or to_input_shape is None:
            return None
        in_f = LayerBridgeFinder.linear_feature_dim(from_output_shape)
        out_f = LayerBridgeFinder.linear_feature_dim(to_input_shape)
        if in_f is None or out_f is None:
            return None
        return in_f, out_f

    @staticmethod
    def find_bridge_res_linear_sizes(
        from_output_shape: tuple[int, ...] | None,
        to_output_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """(in_features, out_features) for a residual bridge Linear."""
        if from_output_shape is None or to_output_shape is None:
            return None
        in_f = LayerBridgeFinder.linear_feature_dim(from_output_shape)
        out_f = LayerBridgeFinder.linear_feature_dim(to_output_shape)
        if in_f is None or out_f is None:
            return None
        return in_f, out_f

    @staticmethod
    def find_equal_conv_output_shapes(
        from_output_shape: tuple[int, ...] | None,
        to_output_shape: tuple[int, ...] | None,
    ) -> bool:
        """True when both shapes are 4-D and identical."""
        return (
            from_output_shape is not None
            and to_output_shape is not None
            and len(from_output_shape) == 4
            and from_output_shape == to_output_shape
        )

    @staticmethod
    def find_conv_before_linear_sizes(
        conv_output_shape: tuple[int, ...] | None,
        linear_input_shape: tuple[int, ...] | None,
        linear_output_shape: tuple[int, ...] | None = None,
        *,
        for_residual: bool = False,
    ) -> tuple[int, int] | None:
        """(in_channels, out) for a conv placed before a linear layer."""
        channels = LayerBridgeFinder.conv_channels(conv_output_shape)
        linear_in = LayerBridgeFinder.linear_feature_dim(linear_input_shape)
        if channels is None or linear_in is None or linear_in % channels != 0:
            return None
        if for_residual:
            linear_out = LayerBridgeFinder.linear_feature_dim(linear_output_shape)
            if linear_out is None:
                return None
            return channels, linear_out
        return channels, channels

    @staticmethod
    def find_res_conv_before_linear_sizes(
        conv_output_shape: tuple[int, ...] | None,
        linear_input_shape: tuple[int, ...] | None,
        linear_output_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """Residual variant of find_conv_before_linear_sizes."""
        return LayerBridgeFinder.find_conv_before_linear_sizes(
            conv_output_shape, linear_input_shape, linear_output_shape, for_residual=True,
        )

    @staticmethod
    def find_seq_conv_before_linear_sizes(
        conv_output_shape: tuple[int, ...] | None,
        linear_input_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """Sequential variant of find_conv_before_linear_sizes."""
        return LayerBridgeFinder.find_conv_before_linear_sizes(
            conv_output_shape, linear_input_shape, for_residual=False,
        )

    @staticmethod
    def find_seq_linear_after_conv_sizes(
        conv_output_shape: tuple[int, ...] | None,
        linear_input_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """(in_features, out_features) for a linear on conv->...->linear path."""
        if LayerBridgeFinder.conv_channels(conv_output_shape) is None:
            return None
        linear_in = LayerBridgeFinder.linear_feature_dim(linear_input_shape)
        if linear_in is None:
            return None
        return linear_in, linear_in

    @staticmethod
    def find_seq_conv_bridge_channels(
        from_output_shape: tuple[int, ...] | None,
        to_input_shape: tuple[int, ...] | None,
    ) -> int | None:
        """Channel count for a sequential conv bridge between two 4-D layers."""
        if from_output_shape is None or from_output_shape != to_input_shape:
            return None
        return LayerBridgeFinder.conv_channels(from_output_shape)
