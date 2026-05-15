from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.actions.utils.model_analyser import get_layer_module, module_dependency_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddResLayer(Action):

    SUPPORTED_MODULES = (nn.Linear,)

    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])
    
    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule, layer_types: List[Layer_Type] = Layer_Type) -> List[Action]:
        """Residual add is ``dst + proj(src)`` on *outputs*; proj must be
        ``Linear(layer_from.out_features, layer_to.out_features)`` (see ``add_new_residual_layer``).
        """
        actions : List[Action] = []
        name_prefix = "res_linear_"
        pairs = module_dependency_pairs(model)
        for layer_from_id, layer_to_id in pairs:
            layer_from = get_layer_module(layer_from_id, model)
            layer_to = get_layer_module(layer_to_id, model)

            if not isinstance(layer_from, AddResLayer.SUPPORTED_MODULES):
                logger.debug("layer_from: %s is not a supported module", layer_from)
                continue
            if not isinstance(layer_to, AddResLayer.SUPPORTED_MODULES):
                logger.debug("layer_to: %s is not a supported module", layer_to)
                continue

            layer_from_out_features = layer_from.out_features
            layer_to_out_features = layer_to.out_features
            for type in layer_types:
                name = unique_call_module_name(name_prefix + type.name, model)
                layer = LinearFactory.create_linear(
                    layer_from_out_features, layer_to_out_features, type
                )
                actions.append(AddResLayer([layer_from_id, layer_to_id, layer, name]))
        return actions
    
    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "