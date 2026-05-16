from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.actions.utils.layer_analyser import LayerBridgeFinder, LayerShapeAnalyser
from growingnn.actions.utils.model_analyser import module_sequential_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_seq_layer
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddSeqLayer(Action):

    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_seq_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)
        in_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm)
        actions: List[Action] = []
        for layer_from_id, layer_to_id in module_sequential_pairs(gm):
            s_out = out_shapes.get(layer_from_id)
            s_in = in_shapes.get(layer_to_id)
            sizes = LayerBridgeFinder.find_bridge_linear_sizes(s_out, s_in)
            if sizes is not None:
                name = unique_call_module_name("seq_linear", gm)
                layer = LinearFactory.create_linear(sizes[0], sizes[1], Layer_Type.EYE)
                logger.debug(
                    "AddSeqLayer %s -> %s: Linear(%d, %d) out=%s in=%s",
                    layer_from_id,
                    layer_to_id,
                    sizes[0],
                    sizes[1],
                    s_out,
                    s_in,
                )
                actions.append(AddSeqLayer([layer_from_id, layer_to_id, layer, name]))
                continue
            conv_linear_sizes = LayerBridgeFinder.find_seq_linear_after_conv_sizes(s_out, s_in)
            if conv_linear_sizes is None:
                logger.debug("AddSeqLayer skip %s -> %s", layer_from_id, layer_to_id)
                continue
            name = unique_call_module_name("seq_linear", gm)
            layer = LinearFactory.create_linear(
                conv_linear_sizes[0],
                conv_linear_sizes[1],
                Layer_Type.EYE,
            )
            logger.debug(
                "AddSeqLayer %s -> %s: conv->linear (%d, %d) out=%s in=%s",
                layer_from_id,
                layer_to_id,
                conv_linear_sizes[0],
                conv_linear_sizes[1],
                s_out,
                s_in,
            )
            actions.append(AddSeqLayer([layer_from_id, layer_to_id, layer, name]))
        return actions

    def __str__(self):
        return " ( Add Seq Layer Action: " + str(self.params) + " ) "
