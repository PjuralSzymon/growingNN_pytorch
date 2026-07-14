from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    LayerBridgeFinder,
    ModuleResolver, ModelStructureEditor,
)
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddSeqLinearLayer(Action):

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
            sizes = LayerBridgeFinder.find_bridge_linear_sizes(s_out, s_in)
            if sizes is not None:
                if sizes[0] * sizes[1] > config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE:
                    continue
                name = ModuleResolver.unique_call_module_name("seq_linear", gm)
                layer = LinearFactory.create_linear(sizes[0], sizes[1], Layer_Type.EYE)
                logger.debug(
                    "AddSeqLinearLayer %s -> %s: Linear(%d, %d) out=%s in=%s",
                    layer_from_id,
                    layer_to_id,
                    sizes[0],
                    sizes[1],
                    s_out,
                    s_in,
                )
                actions.append(AddSeqLinearLayer([layer_from_id, layer_to_id, layer, name]))
                continue
            conv_linear_sizes = LayerBridgeFinder.find_seq_linear_after_conv_sizes(s_out, s_in)
            if conv_linear_sizes is None:
                logger.debug("AddSeqLinearLayer skip %s -> %s", layer_from_id, layer_to_id)
                continue
            if conv_linear_sizes[0] * conv_linear_sizes[1] > config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE:
                continue
            name = ModuleResolver.unique_call_module_name("seq_linear", gm)
            layer = LinearFactory.create_linear(
                conv_linear_sizes[0],
                conv_linear_sizes[1],
                Layer_Type.EYE,
            )
            logger.debug(
                "AddSeqLinearLayer %s -> %s: conv->linear (%d, %d) out=%s in=%s",
                layer_from_id,
                layer_to_id,
                conv_linear_sizes[0],
                conv_linear_sizes[1],
                s_out,
                s_in,
            )
            actions.append(AddSeqLinearLayer([layer_from_id, layer_to_id, layer, name]))
        return actions

    def __str__(self):
        return " ( Add Seq Linear Layer Action: " + str(self.params) + " ) "
