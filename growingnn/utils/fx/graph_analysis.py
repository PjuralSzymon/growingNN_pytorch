"""Whole-graph read-only queries: module classification, topology, shape propagation, bridge sizing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from growingnn.core.config import EDITABLE_MODULES
from growingnn.core.logger import logger
from growingnn.utils.fx.node_analysis import ModuleResolver, NodeTypeChecker


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
        """All (ancestor, descendant) editable pairs on the forward path (boundaries allowed)."""
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
                if ModuleClassifier.is_editable_module(cur, gm):
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
    def find_flatten_node_on_path_toward_source(
        path_destination_to_source: list[fx.Node],
        gm: fx.GraphModule,
    ) -> fx.Node | None:
        """Return the first flatten node walking a destination-to-source FX path, or None."""
        for node in path_destination_to_source:
            if NodeTypeChecker.is_flatten_node(node, gm):
                return node
        return None

    @staticmethod
    def find_pool_nodes_between_flatten_and_source(
        path_destination_to_source: list[fx.Node],
        flatten_node: fx.Node,
        gm: fx.GraphModule,
    ) -> list[fx.Node]:
        """Return pool nodes between flatten and source (source-ward of flatten on the path)."""
        flatten_index = path_destination_to_source.index(flatten_node)
        pools: list[fx.Node] = []
        for node in path_destination_to_source[flatten_index + 1 :]:
            if NodeTypeChecker.is_pool_node(node, gm):
                pools.append(node)
        return pools

    @staticmethod
    def _sequential_adj(model: nn.Module | fx.GraphModule) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Build predecessor and successor adjacency dicts from sequential pairs."""
        pred: dict[str, list[str]] = {}
        succ: dict[str, list[str]] = {}
        for a, b in dict.fromkeys(GraphStructureQuery.module_sequential_pairs(model)):
            pred.setdefault(b, []).append(a)
            succ.setdefault(a, []).append(b)
        return pred, succ


class GraphConnectivity:
    """Reachability and dangling-branch diagnostics for FX graphs."""

    @staticmethod
    def nodes_reachable_from_output(gm: fx.GraphModule) -> set[fx.Node]:
        """Return every FX node on a path to the graph output."""
        output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
        if not output_nodes:
            return set(gm.graph.nodes)
        live: set[fx.Node] = set()
        stack = list(output_nodes)
        while stack:
            node = stack.pop()
            if node in live:
                continue
            live.add(node)
            stack.extend(node.all_input_nodes)
        return live

    @staticmethod
    def get_output_module_id(gm: fx.GraphModule) -> str | None:
        """Return the call_module target wired into the graph output node."""
        output = next((node for node in gm.graph.nodes if node.op == "output"), None)
        if output is None or not output.args:
            return None
        arg = output.args[0]
        if isinstance(arg, fx.Node) and arg.op == "call_module":
            return str(arg.target)
        return None

    @staticmethod
    def get_input_module_ids(gm: fx.GraphModule) -> list[str]:
        """Return call_module targets fed directly by graph placeholders."""
        placeholders = {node for node in gm.graph.nodes if node.op == "placeholder"}
        ids: list[str] = []
        for node in gm.graph.nodes:
            if node.op != "call_module":
                continue
            if placeholders.intersection(node.all_input_nodes):
                ids.append(str(node.target))
        return list(dict.fromkeys(ids))

    @staticmethod
    def dangling_leaf_nodes(gm: fx.GraphModule) -> list[fx.Node]:
        """Return FX nodes with no users that are not placeholders or output."""
        return [
            node for node in gm.graph.nodes
            if len(node.users) == 0 and node.op not in ("placeholder", "output")
        ]

    @staticmethod
    def unreachable_module_ids(gm: fx.GraphModule) -> list[str]:
        """Return call_module targets that do not reach the graph output."""
        live = GraphConnectivity.nodes_reachable_from_output(gm)
        return [
            str(node.target) for node in gm.graph.nodes
            if node.op == "call_module" and node not in live
        ]

    @staticmethod
    def live_module_ids(gm: fx.GraphModule) -> list[str]:
        """Return call_module targets on a path to the graph output."""
        live = GraphConnectivity.nodes_reachable_from_output(gm)
        return [
            str(node.target) for node in gm.graph.nodes
            if node.op == "call_module" and node in live
        ]

    @staticmethod
    def is_connected_to_output(gm: fx.GraphModule) -> bool:
        """True when every call_module reaches the output and there are no dangling leaves."""
        return (
            not GraphConnectivity.dangling_leaf_nodes(gm)
            and not GraphConnectivity.unreachable_module_ids(gm)
        )

    @staticmethod
    def explain_connectivity(gm: fx.GraphModule) -> list[str]:
        """Return human-readable connectivity problems for logging and tests."""
        issues: list[str] = []
        input_ids = GraphConnectivity.get_input_module_ids(gm)
        output_id = GraphConnectivity.get_output_module_id(gm)
        if len(input_ids) != 1:
            issues.append(f"expected 1 input module, got {len(input_ids)}: {input_ids}")
        if output_id is None:
            issues.append("graph output is not wired to a call_module")
        dangling = GraphConnectivity.dangling_leaf_nodes(gm)
        if dangling:
            issues.append(
                "dangling leaves: "
                + ", ".join(
                    f"{node.op}:{getattr(node, 'target', node.name)}"
                    for node in dangling
                )
            )
        unreachable = GraphConnectivity.unreachable_module_ids(gm)
        if unreachable:
            issues.append(f"unreachable modules: {unreachable}")
        live = GraphConnectivity.live_module_ids(gm)
        if live:
            issues.append(f"live path modules: {live}")
        return issues


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
    def make_probe(gm: fx.GraphModule, input_shape: tuple[int, ...]) -> torch.Tensor:
        """Build a random probe tensor for ShapeProp."""
        try:
            p0 = next(gm.parameters())
            device, dtype = p0.device, p0.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32
        return torch.randn(*input_shape, device=device, dtype=dtype)

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
        *,
        input_shape: tuple[int, ...] | None = None,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        """Run ShapeProp and return (output_shapes, input_shapes) dicts keyed by module target."""
        outputs: dict[str, tuple[int, ...]] = {}
        inputs: dict[str, tuple[int, ...]] = {}
        if example is not None:
            probe = example
        elif input_shape is not None:
            probe = LayerShapeAnalyser.make_probe(gm, input_shape)
        else:
            raise ValueError("collect_layer_shapes requires example or input_shape")
        was_training = gm.training
        gm.eval()
        try:
            ShapeProp(gm).propagate(probe)
        finally:
            gm.train(was_training)
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
        *,
        input_shape: tuple[int, ...] | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Output activation shapes for every call_module node."""
        return LayerShapeAnalyser.collect_layer_shapes(gm, example, input_shape=input_shape)[0]

    @staticmethod
    def get_layer_input_shapes(
        gm: fx.GraphModule,
        example: torch.Tensor | None = None,
        *,
        input_shape: tuple[int, ...] | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Input activation shapes for every call_module node."""
        return LayerShapeAnalyser.collect_layer_shapes(gm, example, input_shape=input_shape)[1]


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
    def linear_feature_dim(shape: tuple[int, ...] | None) -> int | None:
        """Feature dimension from a 2-D (batch, features) shape."""
        if shape is None:
            return None
        if len(shape) != 2:
            return None
        features = int(shape[1])
        return features if features > 0 else None

    @staticmethod
    def conv_channels(shape: tuple[int, ...] | None) -> int | None:
        """Channel dimension from a 4-D (batch, channels, H, W) shape."""
        if shape is None:
            return None
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

    @staticmethod
    def _normalize_to_height_width_pair(
        value: int | tuple[int, ...] | list[int],
    ) -> tuple[int, int]:
        """Turn an int or (height, width) size into a (height, width) pair for size formulas."""
        if isinstance(value, int):
            return value, value
        return int(value[0]), int(value[1])

    @staticmethod
    def get_convolution_output_height_and_width(
        input_height: int,
        input_width: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
    ) -> tuple[int, int] | None:
        """Return Conv2d output (height, width), or None when the result is invalid."""
        kernel_h, kernel_w = LayerBridgeFinder._normalize_to_height_width_pair(kernel_size)
        stride_h, stride_w = LayerBridgeFinder._normalize_to_height_width_pair(stride)
        pad_h, pad_w = LayerBridgeFinder._normalize_to_height_width_pair(padding)
        dil_h, dil_w = LayerBridgeFinder._normalize_to_height_width_pair(dilation)
        out_h = (input_height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (input_width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
        if out_h < 1 or out_w < 1:
            return None
        return out_h, out_w

    @staticmethod
    def get_flatten_feature_count_after_convolution_and_pools(
        insert_site_channels: int,
        insert_site_height: int,
        insert_site_width: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        pools_after_insert: list[tuple[str, dict]],
    ) -> int | None:
        """Return C*H*W after a new conv then optional pools before flatten."""
        del insert_site_channels
        spatial = LayerBridgeFinder.get_convolution_output_height_and_width(
            insert_site_height,
            insert_site_width,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        if spatial is None:
            return None
        height, width = spatial
        channels = out_channels
        for kind, kwargs in pools_after_insert:
            if kind.startswith("adaptive"):
                height, width = LayerBridgeFinder._normalize_to_height_width_pair(
                    kwargs["output_size"],
                )
            else:
                pool_kernel = kwargs["kernel_size"]
                pool_stride = kwargs.get("stride")
                if pool_stride is None:
                    pool_stride = pool_kernel
                spatial = LayerBridgeFinder.get_convolution_output_height_and_width(
                    height,
                    width,
                    pool_kernel,
                    stride=pool_stride,
                    padding=kwargs.get("padding", 0),
                    dilation=kwargs.get("dilation", 1),
                )
                if spatial is None:
                    return None
                height, width = spatial
        return channels * height * width
