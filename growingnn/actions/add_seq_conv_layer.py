from typing import List

import torch.nn as nn

from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    FeatureAxisConv2dAdapter,
    GraphStructureQuery,
    LayerBridgeFinder,
    LayerShapeAnalyser,
    ModuleResolver,
    ModelStructureEditor,
    analyze_feature_tensor,
    pair_traversal_decision,
)

from growingnn.core.logger import logger
from .action import Action


class AddSeqConvLayer(Action):
    """
    params: [from_id, to_id, layer, name, insert_before_flatten]
    insert_before_flatten is set in generate_all_actions (True for conv→linear stems).
    """

    def _execute(self, traced: TracedModel):
        if len(self.params) > 4 and self.params[4] == "fx_edge":
            source_name, destination_name, layer, name = self.params[:4]
            nodes = {node.name: node for node in traced.gm.graph.nodes}
            source = nodes.get(source_name)
            destination = nodes.get(destination_name)
            if source is None or destination is None:
                raise ValueError(
                    f"FX convolution action refers to missing nodes {source_name!r} -> {destination_name!r}"
                )
            ModelStructureEditor.insert_module_on_edge(
                traced.gm,
                source,
                destination,
                layer,
                name,
                use_path=self.params[5] if len(self.params) > 5 else None,
            )
            return
        from_id, to_id, layer, name, insert_before_flatten = self.params[:5]
        add_batchnorm_and_relu = bool(self.params[5]) if len(self.params) > 5 else False
        before_global_pool = bool(self.params[6]) if len(self.params) > 6 else True
        if insert_before_flatten:
            ModelStructureEditor.add_new_seq_layer_before_flatten(
                traced.gm, from_id, to_id, layer, name,
                add_batchnorm_and_relu=add_batchnorm_and_relu,
                before_global_pool=before_global_pool,
            )
            return
        ModelStructureEditor.add_new_seq_layer(
            traced.gm, from_id, to_id, layer, name, add_batchnorm_and_relu=add_batchnorm_and_relu,
        )

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
    def generate_all_actions(
        traced: TracedModel,
        add_batchnorm_and_relu: bool = False,
        before_global_pool: bool = True,
        *,
        feature_axis: int | None = None,
    ) -> List[Action]:
        gm = traced.gm
        out_shapes, in_shapes = traced.shapes()
        actions: List[Action] = []
        pair_decision = pair_traversal_decision(gm)
        if not pair_decision.use_legacy_pairs:
            logger.info(
                "AddSeqConvLayer using rank-4 FX channel seams instead of exhaustive module-pair search: %s",
                "; ".join(pair_decision.reasons),
            )
            if feature_axis is None:
                logger.info(
                    "AddSeqConvLayer produced no FX candidates: provide feature_axis explicitly for rank-4 channel tensors"
                )
                return actions
            observed: set[tuple[object, object]] = set()
            for site in traced.action_sites(include_rejected=False):
                source, destination = site.edge.source, site.edge.destination
                if (
                    source is None
                    or destination is None
                    or (source, destination) in observed
                    or not site.meaningful_seam
                ):
                    continue
                observed.add((source, destination))
                feature = analyze_feature_tensor(
                    site.interface,
                    feature_axis={4: feature_axis},
                    supported_ranks=(4,),
                )
                if not feature.eligible:
                    logger.debug(
                        "AddSeqConvLayer FX skip %s -> %s: %s",
                        source.name, destination.name, feature.rejection_reason,
                    )
                    continue
                if feature.feature_axis == 0:
                    logger.debug(
                        "AddSeqConvLayer FX skip %s -> %s: feature axis cannot be the batch axis",
                        source.name, destination.name,
                    )
                    continue
                if feature.dtype is None or not feature.dtype.is_floating_point:
                    logger.debug(
                        "AddSeqConvLayer FX skip %s -> %s: dtype must be floating",
                        source.name, destination.name,
                    )
                    continue
                device = feature.device or "cpu"
                adapter = FeatureAxisConv2dAdapter(
                    feature.features,
                    feature_axis=feature.feature_axis,
                    residual=True,
                ).to(device=device, dtype=feature.dtype)
                use_paths = site.edge.use_paths or (None,)
                for use_path in use_paths:
                    name = ModuleResolver.unique_call_module_name("seq_conv", gm)
                    layer = adapter if len(use_paths) == 1 else FeatureAxisConv2dAdapter(
                        feature.features,
                        feature_axis=feature.feature_axis,
                        residual=True,
                    ).to(device=device, dtype=feature.dtype)
                    params = [source.name, destination.name, layer, name, "fx_edge"]
                    if use_path is not None:
                        params.append(use_path)
                    actions.append(AddSeqConvLayer(params))
            return actions
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
            params = [layer_from_id, layer_to_id, layer, name, insert_before_flatten]
            if add_batchnorm_and_relu or not before_global_pool:
                params.append(bool(add_batchnorm_and_relu))
                params.append(bool(before_global_pool))
            actions.append(AddSeqConvLayer(params))
        return actions

    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "
