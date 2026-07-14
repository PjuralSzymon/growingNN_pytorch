from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import ConvFactory
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    LayerBridgeFinder,
    ModuleResolver, ModelStructureEditor,
)
from growingnn.core.logger import logger
from .action import Action


class AddSeqConvLayer(Action):

    def _execute(self, traced: TracedModel):
        ModelStructureEditor.add_new_seq_layer(traced.gm, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

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
        return actions

    def __str__(self):
        return " ( Add Seq Conv Layer Action: " + str(self.params) + " ) "
