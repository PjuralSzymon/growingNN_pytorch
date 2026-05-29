from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.utils.fx import (
    LayerBridgeFinder, LayerShapeAnalyser,
    ModuleResolver, GraphStructureQuery, ModelStructureEditor,
)
from growingnn.core.logger import logger
from .action import Action


class AddSeqConvLayer(Action):

    def execute(self, model: nn.Module | fx.GraphModule):
        ModelStructureEditor.add_new_seq_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)
        in_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm)
        actions: List[Action] = []
        for layer_from_id, layer_to_id in GraphStructureQuery.module_sequential_pairs(gm):
            s_out = out_shapes.get(layer_from_id)
            s_in = in_shapes.get(layer_to_id)
            layer_from = ModuleResolver.get_layer_module(layer_from_id, model)
            channels = LayerBridgeFinder.find_seq_conv_bridge_channels(s_out, s_in)
            if channels is not None:
                name = ModuleResolver.unique_call_module_name("seq_conv", gm)
                layer = ConvFactory.create_eye_conv(
                    channels,
                    channels,
                    layer_from.kernel_size,
                    stride=1,
                    padding=layer_from.padding,
                )
                logger.debug("AddSeqConvLayer %s -> %s: eye conv %d out=%s in=%s", layer_from_id, layer_to_id, channels, s_out, s_in)
                actions.append(AddSeqConvLayer([layer_from_id, layer_to_id, layer, name]))
                continue
            # sizes = LayerBridgeFinder.find_seq_conv_before_linear_sizes(s_out, s_in)
            # if sizes is None:
            #     continue
            # name = unique_call_module_name("seq_conv", gm)
            # layer = ConvFactory.create_zero_conv_before_linear(
            #     sizes[0],
            #     sizes[1],
            #     layer_from.kernel_size,
            #     stride=1,
            #     padding=layer_from.padding,
            # )
            # logger.debug("AddSeqConvLayer %s -> %s: conv->linear (%d,%d) conv=%s lin_in=%s", layer_from_id, layer_to_id, sizes[0], sizes[1], s_out, s_in)
            # actions.append(AddSeqConvLayer([layer_from_id, layer_to_id, layer, name]))
        return actions

    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "
