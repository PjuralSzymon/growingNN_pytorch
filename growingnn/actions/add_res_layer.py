
from torch import fx, nn

from growingnn.actions.utils.model_analyser import module_dependency_pairs
from growingnn.actions.utils.model_transformations import add_new_residual_layer
from .action import Action, Layer_Type


class AddResLayer(Action):

    def execute(self, model: nn.Module | fx.GraphModule):
        add_new_residual_layer(model, self.params[0], self.params[1], self.params[2])
    
    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule):
        actions = []
        pairs = module_dependency_pairs(model)
        for layer_from_id, layer_to_id in pairs:
            layer_to = getattr(model, layer_to_id, None)
            if not isinstance(layer_to, nn.modules.conv._ConvNd):
                for layer_type in Layer_Type:
                    actions.append(AddResLayer([layer_from_id, layer_to_id, layer_type]))
        return actions
    
    def __str__(self):
        return " ( Add Res Layer Action: " + str(self.params) + " ) "