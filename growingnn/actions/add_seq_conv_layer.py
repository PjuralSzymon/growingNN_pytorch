from typing import List

import torch.nn as nn

from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    GraphStructureQuery,
    LayerBridgeFinder,
    LayerShapeAnalyser,
    ModuleResolver,
    ModelStructureEditor,
)

from growingnn.core.logger import logger
from .action import Action


class AddSeqConvLayer(Action):
    """
    params: [from_id, to_id, layer, name, insert_before_flatten]
    insert_before_flatten is set in generate_all_actions (True for conv→linear stems).
    """

    def _execute(self, traced: TracedModel):
        from_id, to_id, layer, name, insert_before_flatten = self.params
        if insert_before_flatten:
            ModelStructureEditor.add_new_seq_layer_before_flatten(
                traced.gm, from_id, to_id, layer, name,
            )
            return
        ModelStructureEditor.add_new_seq_layer(traced.gm, from_id, to_id, layer, name)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def _get_eye_conv_kernel_padding_pairs(
        source_conv: nn.Conv2d,
    ) -> list[tuple[int | tuple[int, ...], int | tuple[int, ...]]]:
        """Return (kernel_size, padding) pairs: source first, then (1, 0); dedupe identical."""
        pairs = [(source_conv.kernel_size, source_conv.padding), (1, 0)]
        unique: list[tuple[int | tuple[int, ...], int | tuple[int, ...]]] = []
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for kernel_size, padding in pairs:
            key = (
                LayerBridgeFinder._normalize_to_height_width_pair(kernel_size),
                LayerBridgeFinder._normalize_to_height_width_pair(padding),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append((kernel_size, padding))
        return unique

    @staticmethod
    def is_before_flatten_insert(
        traced: TracedModel,
        layer_from_id: str,
        layer_to_id: str,
    ) -> bool:
        """True for Conv2d→Linear with flatten, ≥1 pool before flatten, and a 4-D producer."""
        gm = traced.gm
        layer_from = ModuleResolver.get_layer_module(layer_from_id, gm)
        layer_to = ModuleResolver.get_layer_module(layer_to_id, gm)
        if not isinstance(layer_from, nn.Conv2d) or not isinstance(layer_to, nn.Linear):
            return False
        _, in_shapes = traced.shapes()
        if LayerBridgeFinder.linear_feature_dim(in_shapes.get(layer_to_id)) is None:
            return False
        src_node = ModuleResolver.find_call_module(gm.graph.nodes, layer_from_id)
        dst_node = ModuleResolver.find_call_module(gm.graph.nodes, layer_to_id)
        if src_node is None or dst_node is None:
            return False
        path = LayerBridgeFinder.path_dst_to_src(dst_node, src_node)
        if path is None:
            return False
        flatten_node = GraphStructureQuery.find_flatten_node_on_path_toward_source(path, gm)
        if flatten_node is None:
            return False
        flatten_index = path.index(flatten_node)
        # Destination must be the first Linear after flatten (no intervening Linear).
        for node in path[1:flatten_index]:
            if node.op != "call_module":
                continue
            module = ModuleResolver.get_layer_module(node, gm)
            if isinstance(module, nn.Linear):
                return False
        if not GraphStructureQuery.find_pool_nodes_between_flatten_and_source(path, flatten_node, gm):
            return False
        if flatten_index + 1 >= len(path):
            return False
        shape = LayerShapeAnalyser.node_shape(path[flatten_index + 1])
        return shape is not None and len(shape) == 4

    @staticmethod
    def _get_before_flatten_insert_shape(
        traced: TracedModel,
        layer_from_id: str,
        layer_to_id: str,
    ) -> tuple[int, int, int, int]:
        """
        Return (channels, height, width, linear_in_features) at the node before flatten.

        Call only when is_before_flatten_insert is True.
        """
        gm = traced.gm
        _, in_shapes = traced.shapes()
        linear_in_features = LayerBridgeFinder.linear_feature_dim(in_shapes.get(layer_to_id))
        src_node = ModuleResolver.find_call_module(gm.graph.nodes, layer_from_id)
        dst_node = ModuleResolver.find_call_module(gm.graph.nodes, layer_to_id)
        path = LayerBridgeFinder.path_dst_to_src(dst_node, src_node)
        flatten_node = GraphStructureQuery.find_flatten_node_on_path_toward_source(path, gm)
        _, channels, height, width = LayerShapeAnalyser.node_shape(
            path[path.index(flatten_node) + 1],
        )
        return channels, height, width, linear_in_features

    @staticmethod
    def get_eye_convolution_shape_for_before_flatten(
        traced: TracedModel,
        layer_from_id: str,
        layer_to_id: str,
    ) -> tuple[int, int | tuple[int, ...], int | tuple[int, ...]] | None:
        """
        Return (out_channels, kernel_size, padding) for a sequential eye conv before flatten.

        Call only when is_before_flatten_insert is True. Returns None when no eye shape matches.
        """
        channels, height, width, linear_in_features = (
            AddSeqConvLayer._get_before_flatten_insert_shape(traced, layer_from_id, layer_to_id)
        )
        source_conv = ModuleResolver.get_layer_module(layer_from_id, traced.gm)
        for kernel_size, padding in AddSeqConvLayer._get_eye_conv_kernel_padding_pairs(source_conv):
            feature_count = LayerBridgeFinder.get_flatten_feature_count_after_convolution_and_pools(
                channels,
                height,
                width,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=1,
                pools_after_insert=[],
            )
            if feature_count is not None and feature_count == linear_in_features:
                return channels, kernel_size, padding
        return None

    @staticmethod
    def generate_all_actions(traced: TracedModel) -> List[Action]:
        gm = traced.gm
        out_shapes, in_shapes = traced.shapes()
        actions: List[Action] = []
        for layer_from_id, layer_to_id in traced.sequential_pairs():
            s_out = out_shapes.get(layer_from_id)
            s_in = in_shapes.get(layer_to_id)
            layer_from = ModuleResolver.get_layer_module(layer_from_id, gm)
            channels = LayerBridgeFinder.find_seq_conv_bridge_channels(s_out, s_in)

            if channels is not None:
                # Matching 4-D activations: sequential conv between two conv-shaped edges.
                layer = ConvFactory.create_eye_conv(
                    channels, channels, layer_from.kernel_size, stride=1, padding=layer_from.padding,
                )
                insert_before_flatten = False
                logger.debug(
                    "AddSeqConvLayer %s -> %s: matching 4-D eye conv %d out=%s in=%s",
                    layer_from_id, layer_to_id, channels, s_out, s_in,
                )
            elif AddSeqConvLayer.is_before_flatten_insert(traced, layer_from_id, layer_to_id):
                # Conv→…→flatten→linear: eye conv immediately before flatten.
                eye_shape = AddSeqConvLayer.get_eye_convolution_shape_for_before_flatten(
                    traced, layer_from_id, layer_to_id,
                )
                if eye_shape is None:
                    logger.debug(
                        "AddSeqConvLayer skip %s -> %s: no matching before-flatten eye shape",
                        layer_from_id, layer_to_id,
                    )
                    continue
                out_channels, kernel_size, padding = eye_shape
                layer = ConvFactory.create_eye_conv(
                    out_channels, out_channels, kernel_size, stride=1, padding=padding,
                )
                insert_before_flatten = True
                logger.debug(
                    "AddSeqConvLayer %s -> %s: before-flatten eye conv %d out=%s in=%s",
                    layer_from_id, layer_to_id, out_channels, s_out, s_in,
                )
            else:
                continue

            name = ModuleResolver.unique_call_module_name("seq_conv", gm)
            actions.append(
                AddSeqConvLayer([layer_from_id, layer_to_id, layer, name, insert_before_flatten]),
            )
        return actions

    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "
