from typing import Iterable, List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.actions.utils.layer_analyser import LayerBridgeFinder, LayerShapeAnalyser
from growingnn.actions.utils.model_analyser import module_dependency_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddResLayer(Action):

    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        model: nn.Module | fx.GraphModule,
        layer_types: Iterable[Layer_Type] = Layer_Type,
    ) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        out_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)
        actions: List[Action] = []
        for layer_from_id, layer_to_id in module_dependency_pairs(gm):
            sizes = LayerBridgeFinder.find_bridge_res_linear_sizes(
                out_shapes.get(layer_from_id),
                out_shapes.get(layer_to_id),
            )
            if sizes is None:
                logger.debug("AddResLayer skip %s -> %s", layer_from_id, layer_to_id)
                continue
            for layer_type in layer_types:
                name = unique_call_module_name(f"res_linear_{layer_type.name}", gm)
                layer = LinearFactory.create_linear(sizes[0], sizes[1], layer_type)
                logger.debug("AddResLayer %s -> %s: Linear(%d, %d) %s out=%s/%s", layer_from_id, layer_to_id, sizes[0], sizes[1], layer_type.name, out_shapes.get(layer_from_id), out_shapes.get(layer_to_id))
                actions.append(AddResLayer([layer_from_id, layer_to_id, layer, name]))
        return actions

    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "
