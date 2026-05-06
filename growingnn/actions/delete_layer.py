
from typing import List

from torch import fx, nn
from .action import Action, Layer_Type

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.actions.utils.model_analyser import get_all_hidden_modules, module_sequential_pairs
from growingnn.actions.utils.name_factory import unique_call_module_name
from growingnn.actions.utils.model_transformations import add_new_residual_layer, add_new_seq_layer, delete_layer
from .action import Action, Layer_Type

class DelLayer(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        delete_layer(model, self.params[0])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        actions : List[Action] = []
        layers = get_all_hidden_modules(model)
        for layer_id in layers:
            actions.append(DelLayer([layer_id]))
        return actions


    def __str__(self):
        return " ( Delete Layer Action: " + str(self.params) + " ) "