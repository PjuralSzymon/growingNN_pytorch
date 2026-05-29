from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.utils.fx import (
    LayerBridgeFinder, LayerShapeAnalyser,
    ModuleResolver, GraphStructureQuery, ModelStructureEditor,
)
from growingnn.core.logger import logger
from .action import Action


class AddResConvLayer(Action):

    def execute(self, model: nn.Module | fx.GraphModule):
        ModelStructureEditor.add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)
        in_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm)
        actions: List[Action] = []
        for layer_from_id, layer_to_id in GraphStructureQuery.module_dependency_pairs(gm):
            s_from = out_shapes.get(layer_from_id)
            s_to = out_shapes.get(layer_to_id)
            if LayerBridgeFinder.find_equal_conv_output_shapes(s_from, s_to):
                name = ModuleResolver.unique_call_module_name("res_conv_", gm)
                layer_from = ModuleResolver.get_layer_module(layer_from_id, model)
                layer_to = ModuleResolver.get_layer_module(layer_to_id, model)
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
            layer_from = ModuleResolver.get_layer_module(layer_from_id, model)
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
