"""Layer shapes (``ShapeProp``), bridge sizing, and lightweight FX-graph queries."""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from growingnn.actions.utils.model_analyser import get_layer_module

PASSTHROUGH_MODULES = (nn.Dropout, nn.Identity, nn.ReLU, nn.LeakyReLU,
                       nn.GELU, nn.SiLU, nn.Tanh, nn.ELU, nn.Sigmoid,
                       nn.BatchNorm1d, nn.BatchNorm2d,
                       nn.MaxPool2d, nn.AvgPool2d,
                       nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)
PASSTHROUGH_FUNCTIONS = frozenset({
    F.relu, F.gelu, F.silu, F.tanh, F.elu, F.sigmoid,
    torch.relu, torch.sigmoid, torch.tanh,
})


def is_passthrough(gm: fx.GraphModule, n: fx.Node) -> bool:
    """True for nodes that forward tensors without changing their shape."""
    return (n.op == "call_function" and n.target in PASSTHROUGH_FUNCTIONS) or \
           (n.op == "call_module" and isinstance(get_layer_module(n.target, gm), PASSTHROUGH_MODULES))


def is_fork(n: fx.Node) -> bool:
    return len(n.users) > 1


def is_add(n: fx.Node) -> bool:
    return n.op == "call_function" and n.target == operator.add


def node_output_width(gm: fx.GraphModule, n: fx.Node) -> int | None:
    """Output channel width reading live module attributes, walking through passthroughs and adds."""
    if n.op == "call_module":
        m = get_layer_module(n.target, gm)
        if isinstance(m, nn.Linear): return m.out_features
        if isinstance(m, nn.Conv2d): return m.out_channels
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): return m.num_features
    if (is_passthrough(gm, n) or is_add(n)) and n.all_input_nodes:
        return node_output_width(gm, n.all_input_nodes[0])
    return None


def inputs_match_width(gm: fx.GraphModule, n: fx.Node, w: int) -> bool:
    """True when every input to node n has output width w."""
    return bool(n.all_input_nodes) and all(node_output_width(gm, i) == w for i in n.all_input_nodes)


def all_sites_match_width(gm: fx.GraphModule, module_name: str, w: int) -> bool:
    """True when every call_module site for module_name has all inputs at width w."""
    return all(inputs_match_width(gm, n, w)
               for n in gm.graph.nodes if n.op == "call_module" and n.target == module_name)


_RESIZE_SAFE = (nn.Linear,)


def propagation_hits_unsizable(gm: fx.GraphModule, start_node: fx.Node) -> bool:
    """True if forward propagation from start_node would reach an add whose
    sibling branch contains a non-resizable call_module (e.g. Conv2d)."""
    seen: set[str] = set()

    def _check_sibling(node: fx.Node) -> bool:
        """Walk backward into a sibling branch looking for non-sizable call_modules."""
        if node.name in seen:
            return False
        seen.add(node.name)
        if is_passthrough(gm, node):
            return any(_check_sibling(inp) for inp in node.all_input_nodes)
        if node.op == "call_module":
            mod = get_layer_module(node.target, gm)
            if mod is not None and not isinstance(mod, _RESIZE_SAFE):
                return True
            return False
        if node.op == "placeholder":
            return False
        return any(_check_sibling(inp) for inp in node.all_input_nodes)

    def _walk(node: fx.Node) -> bool:
        """Walk forward checking every add node's sibling branches."""
        if node.name in seen:
            return False
        seen.add(node.name)
        for user in node.users:
            if user.op == "output":
                continue
            if is_add(user):
                for inp in user.all_input_nodes:
                    if inp is not node and _check_sibling(inp):
                        return True
            if _walk(user):
                return True
        return False

    return _walk(start_node)


class LayerShapeAnalyser:
    """Run ``ShapeProp`` and read per-layer input/output activation shapes on an ``fx.GraphModule``."""

    @staticmethod
    def node_shape(node: fx.Node) -> tuple[int, ...] | None:
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
        return LayerShapeAnalyser.collect_layer_shapes(gm, example)[0]

    @staticmethod
    def get_layer_input_shapes(
        gm: fx.GraphModule,
        example: torch.Tensor | None = None,
    ) -> dict[str, tuple[int, ...]]:
        return LayerShapeAnalyser.collect_layer_shapes(gm, example)[1]


class LayerBridgeFinder:
    """From activation shapes, decide if a bridge layer fits and what sizes it needs."""

    @staticmethod
    def uniform_activation_shape(
        shapes: list[tuple[int, ...] | None],
    ) -> tuple[int, ...] | None:
        if not shapes:
            return None
        if any(s is None for s in shapes):
            return None
        first = shapes[0]
        if all(s == first for s in shapes):
            return first
        return None

    @staticmethod
    def linear_feature_dim(shape: tuple[int, ...]) -> int | None:
        if len(shape) != 2:
            return None
        features = int(shape[1])
        return features if features > 0 else None

    @staticmethod
    def conv_channels(shape: tuple[int, ...]) -> int | None:
        if len(shape) != 4:
            return None
        channels = int(shape[1])
        return channels if channels > 0 else None

    @staticmethod
    def find_bridge_linear_sizes(
        from_output_shape: tuple[int, ...] | None,
        to_input_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        if from_output_shape is None or to_input_shape is None:
            return None
        in_features = LayerBridgeFinder.linear_feature_dim(from_output_shape)
        out_features = LayerBridgeFinder.linear_feature_dim(to_input_shape)
        if in_features is None or out_features is None:
            return None
        return in_features, out_features

    @staticmethod
    def find_bridge_res_linear_sizes(
        from_output_shape: tuple[int, ...] | None,
        to_output_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        if from_output_shape is None or to_output_shape is None:
            return None
        in_features = LayerBridgeFinder.linear_feature_dim(from_output_shape)
        out_features = LayerBridgeFinder.linear_feature_dim(to_output_shape)
        if in_features is None or out_features is None:
            return None
        return in_features, out_features

    @staticmethod
    def find_equal_conv_output_shapes(
        from_output_shape: tuple[int, ...] | None,
        to_output_shape: tuple[int, ...] | None,
    ) -> bool:
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
        return LayerBridgeFinder.find_conv_before_linear_sizes(
            conv_output_shape,
            linear_input_shape,
            linear_output_shape,
            for_residual=True,
        )

    @staticmethod
    def find_seq_conv_before_linear_sizes(
        conv_output_shape: tuple[int, ...] | None,
        linear_input_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        return LayerBridgeFinder.find_conv_before_linear_sizes(
            conv_output_shape, linear_input_shape, for_residual=False
        )

    @staticmethod
    def find_seq_linear_after_conv_sizes(
        conv_output_shape: tuple[int, ...] | None,
        linear_input_shape: tuple[int, ...] | None,
    ) -> tuple[int, int] | None:
        """``(in_features, out_features)`` for a linear on conv->…->linear (pool/flatten stay in the FX path)."""
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
        if from_output_shape is None or from_output_shape != to_input_shape:
            return None
        return LayerBridgeFinder.conv_channels(from_output_shape)
