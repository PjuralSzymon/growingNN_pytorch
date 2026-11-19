from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    FeatureAxisConv2dAdapter,
    GraphRegionAnalyzer,
    LayerBridgeFinder,
    ModuleResolver, ModelStructureEditor,
    analyze_feature_tensor,
    pair_traversal_decision,
)
from growingnn.utils.fx.sum_nodes import is_sum_node
from growingnn.core.logger import logger
from .action import Action


class AddResConvLayer(Action):

    def _execute(self, traced: TracedModel):
        if len(self.params) > 4 and self.params[4] == "fx_residual":
            source_name, destination_name, layer, name = self.params[:4]
            nodes = {node.name: node for node in traced.gm.graph.nodes}
            source = nodes.get(source_name)
            destination = nodes.get(destination_name)
            if source is None or destination is None:
                raise ValueError(
                    f"FX convolution residual refers to missing nodes {source_name!r} -> {destination_name!r}"
                )
            ModelStructureEditor.add_residual_branch_on_nodes(
                traced.gm, destination, source, layer, name
            )
            return
        ModelStructureEditor.add_new_residual_layer(traced.gm, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        *,
        feature_axis: int | None = None,
    ) -> List[Action]:
        gm = traced.gm
        out_shapes, in_shapes = traced.shapes()
        actions: List[Action] = []
        pair_decision = pair_traversal_decision(gm)
        if not pair_decision.use_legacy_pairs:
            logger.info(
                "AddResConvLayer using rank-4 FX merge sites instead of exhaustive module-pair search: %s",
                "; ".join(pair_decision.reasons),
            )
            if feature_axis is None:
                logger.info(
                    "AddResConvLayer produced no FX candidates: provide feature_axis explicitly for rank-4 channel tensors"
                )
                return actions
            for merge in gm.graph.nodes:
                if not is_sum_node(merge) or merge.meta.get("growingnn_generated_merge"):
                    continue
                merge_interface = GraphRegionAnalyzer.describe_node_value(merge)
                merge_feature = analyze_feature_tensor(
                    merge_interface,
                    feature_axis={4: feature_axis},
                    supported_ranks=(4,),
                )
                if not merge_feature.eligible:
                    logger.debug(
                        "AddResConvLayer FX skip merge %s: %s",
                        merge.name, merge_feature.rejection_reason,
                    )
                    continue
                if merge_feature.feature_axis == 0:
                    logger.debug("AddResConvLayer FX skip merge %s: feature axis cannot be the batch axis", merge.name)
                    continue
                if merge_feature.dtype is None or not merge_feature.dtype.is_floating_point:
                    logger.debug("AddResConvLayer FX skip merge %s: dtype must be floating", merge.name)
                    continue
                for source in merge.all_input_nodes:
                    if source.op == "get_attr":
                        logger.debug(
                            "AddResConvLayer FX skip %s -> %s: source is a parameter or buffer",
                            source.name, merge.name,
                        )
                        continue
                    source_interface = GraphRegionAnalyzer.describe_node_value(source)
                    source_feature = analyze_feature_tensor(
                        source_interface,
                        feature_axis={4: feature_axis},
                        supported_ranks=(4,),
                    )
                    if not source_feature.eligible:
                        continue
                    if source_feature.feature_axis == 0:
                        continue
                    if (
                        source_interface.shape != merge_interface.shape
                        or source_feature.dtype != merge_feature.dtype
                        or source_feature.device != merge_feature.device
                    ):
                        logger.debug(
                            "AddResConvLayer FX skip %s -> %s: tensor boundary metadata differs",
                            source.name, merge.name,
                        )
                        continue
                    device = source_feature.device or "cpu"
                    adapter = FeatureAxisConv2dAdapter(
                        source_feature.features,
                        feature_axis=source_feature.feature_axis,
                        residual=False,
                    ).to(device=device, dtype=source_feature.dtype)
                    name = ModuleResolver.unique_call_module_name("res_conv_adapter", gm)
                    actions.append(AddResConvLayer([
                        source.name, merge.name, adapter, name, "fx_residual"
                    ]))
            return actions
        for layer_from_id, layer_to_id in traced.dependency_pairs():
            s_from = out_shapes.get(layer_from_id)
            s_to = out_shapes.get(layer_to_id)
            if LayerBridgeFinder.find_equal_conv_output_shapes(s_from, s_to):
                name = ModuleResolver.unique_call_module_name("res_conv_", gm)
                layer_from = ModuleResolver.get_layer_module(layer_from_id, gm)
                layer_to = ModuleResolver.get_layer_module(layer_to_id, gm)
                layer = ConvFactory.create_zero_conv(
                    layer_from.out_channels,
                    layer_to.out_channels,
                    layer_from.kernel_size,
                    stride=1,
                    padding=layer_from.padding,
                )
                logger.debug("AddResConvLayer %s -> %s: conv residual %s", layer_from_id, layer_to_id, s_from)
                actions.append(AddResConvLayer([layer_from_id, layer_to_id, layer, name]))
                continue
            sizes = LayerBridgeFinder.find_conv_before_linear_sizes(
                s_from,
                in_shapes.get(layer_to_id),
                s_to,
                for_residual=True,
            )
            if sizes is None:
                continue
            name = ModuleResolver.unique_call_module_name("res_conv_", gm)
            layer_from = ModuleResolver.get_layer_module(layer_from_id, gm)
            layer = ConvFactory.create_zero_conv_before_linear(
                sizes[0],
                sizes[1],
                layer_from.kernel_size,
                stride=1,
                padding=layer_from.padding,
            )
            logger.debug("AddResConvLayer %s -> %s: conv->linear (%d,%d) conv=%s lin_in=%s lin_out=%s", layer_from_id, layer_to_id, sizes[0], sizes[1], s_from, in_shapes.get(layer_to_id), s_to)
            actions.append(AddResConvLayer([layer_from_id, layer_to_id, layer, name]))
        return actions

    def __str__(self):
        return " ( Add Res Conv Layer Action: " + str(self.params) + " ) "
