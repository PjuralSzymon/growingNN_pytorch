from typing import List

from torch import fx, nn

from growingnn.actions.utils.model_analyser import get_all_hidden_modules, get_input_layers, get_layer_module, get_output_layers, module_sequential_pairs
from growingnn.actions.utils.model_transformations import delete_layer
from .action import Action


def has_same_output_shape(model: nn.Module | fx.GraphModule, input_layers: list[str]) -> bool:
    mods = [get_layer_module(i, model) for i in input_layers]
    return bool(input_layers) and all(isinstance(m, nn.Linear) for m in mods) and len({m.out_features for m in mods}) == 1


def has_same_input_shape(model: nn.Module | fx.GraphModule, output_layers: list[str]) -> bool:
    mods = [get_layer_module(i, model) for i in output_layers]
    return bool(output_layers) and all(isinstance(m, nn.Linear) for m in mods) and len({m.in_features for m in mods}) == 1


def get_common_output_shape(model: nn.Module | fx.GraphModule, input_layers: list[str]) -> int | None:
    if not has_same_output_shape(model, input_layers):
        return None
    return get_layer_module(input_layers[0], model).out_features


def get_common_input_shape(model: nn.Module | fx.GraphModule, output_layers: list[str]) -> int | None:
    if not has_same_input_shape(model, output_layers):
        return None
    return get_layer_module(output_layers[0], model).in_features


class DelLayer(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        delete_layer(model, self.params[0])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        actions: List[Action] = []
        for layer_id in get_all_hidden_modules(model):
            input_layers = get_input_layers(layer_id, model)
            output_layers = get_output_layers(layer_id, model)
            in_w = get_common_output_shape(model, input_layers)
            out_w = get_common_input_shape(model, output_layers)
            if in_w is None or out_w is None or in_w != out_w:
                continue
            actions.append(DelLayer([layer_id]))
        return actions 

    def __str__(self):
        return " ( Delete Layer Action: " + str(self.params) + " ) "
