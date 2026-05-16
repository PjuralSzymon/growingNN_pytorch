"""Layer shapes (``ShapeProp``) and bridge sizing for architecture actions."""

from __future__ import annotations

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp


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
    def find_seq_conv_bridge_channels(
        from_output_shape: tuple[int, ...] | None,
        to_input_shape: tuple[int, ...] | None,
    ) -> int | None:
        if from_output_shape is None or from_output_shape != to_input_shape:
            return None
        return LayerBridgeFinder.conv_channels(from_output_shape)
