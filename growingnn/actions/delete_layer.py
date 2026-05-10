from typing import List

from torch import fx, nn

from growingnn.actions.utils.model_analyser import get_all_hidden_modules, module_sequential_pairs
from growingnn.actions.utils.model_transformations import delete_layer
from .action import Action


def _sequential_adj(model: nn.Module | fx.GraphModule) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    pred: dict[str, list[str]] = {}
    succ: dict[str, list[str]] = {}
    for a, b in dict.fromkeys(module_sequential_pairs(model)):
        pred.setdefault(b, []).append(a)
        succ.setdefault(a, []).append(b)
    return pred, succ


def get_input_layers(layer_id: str, pred: dict[str, list[str]]) -> list[str]:
    return list(pred.get(layer_id, []))


def get_output_layers(layer_id: str, succ: dict[str, list[str]]) -> list[str]:
    return list(succ.get(layer_id, []))


def has_same_output_shape(model: nn.Module | fx.GraphModule, input_layers: list[str]) -> bool:
    mods = [getattr(model, i) for i in input_layers]
    return bool(input_layers) and all(isinstance(m, nn.Linear) for m in mods) and len({m.out_features for m in mods}) == 1


def has_same_input_shape(model: nn.Module | fx.GraphModule, output_layers: list[str]) -> bool:
    mods = [getattr(model, i) for i in output_layers]
    return bool(output_layers) and all(isinstance(m, nn.Linear) for m in mods) and len({m.in_features for m in mods}) == 1


def get_common_output_shape(model: nn.Module | fx.GraphModule, input_layers: list[str]) -> int | None:
    if not has_same_output_shape(model, input_layers):
        return None
    return getattr(model, input_layers[0]).out_features


def get_common_input_shape(model: nn.Module | fx.GraphModule, output_layers: list[str]) -> int | None:
    if not has_same_input_shape(model, output_layers):
        return None
    return getattr(model, output_layers[0]).in_features


class DelLayer(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        delete_layer(model, self.params[0])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        pred, succ = _sequential_adj(model)
        actions: List[Action] = []
        for layer_id in get_all_hidden_modules(model):
            input_layers = get_input_layers(layer_id, pred)
            output_layers = get_output_layers(layer_id, succ)
            in_w = get_common_output_shape(model, input_layers)
            out_w = get_common_input_shape(model, output_layers)
            if in_w is None or out_w is None or in_w != out_w:
                continue
            actions.append(DelLayer([layer_id]))
        return actions 

    def __str__(self):
        return " ( Delete Layer Action: " + str(self.params) + " ) "
