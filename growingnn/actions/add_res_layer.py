from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.actions.utils.model_analyser import module_dependency_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer
from .action import Action, Layer_Type


class AddResLayer(Action):

    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])
    
    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule, layer_types: List[Layer_Type] = Layer_Type) -> List[Action]:
        actions : List[Action] = []
        name_prefix = "res_linear_"
        pairs = module_dependency_pairs(model)
        for layer_from_id, layer_to_id in pairs:
            layer_from = getattr(model, layer_from_id, None)
            layer_to = getattr(model, layer_to_id, None)

            layer_from_out_features = layer_from.out_features
            layer_to_in_features = layer_to.in_features
            if not isinstance(layer_to, nn.modules.conv._ConvNd):
                for type in layer_types:
                    name = unique_call_module_name(name_prefix + type.name, model)
                    layer = LinearFactory.create_linear(layer_from_out_features, layer_to_in_features, type)
                    actions.append(AddResLayer([layer_from_id, layer_to_id, layer, name]))
        return actions
    
    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "