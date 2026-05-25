from typing import List

from torch import fx, nn

from growingnn.actions.utils.model_analyser import get_all_hidden_modules, get_layer_module
from growingnn.actions.utils.model_transformations import _find_call_module
from growingnn.actions.utils.layer_analyser import propagation_hits_unsizable
from growingnn.actions.utils.layer_resize import resize_layer_output
from growingnn.core import config
from .action import Action


def shrink_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, ratio: float) -> fx.GraphModule:
    """Shrink a Linear layer's output by ratio and propagate shapes."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is not nn.Linear")
    new = max(1, int(mod.out_features * ratio))
    if new >= mod.out_features:
        return gm
    return resize_layer_output(gm, layer_id, new)


class DelNeurons(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        layer_id = self.params[0]
        ratio = self.params[1] if len(self.params) > 1 else config.DEFAULT_NEURONS_SHRINK_RATIO
        shrink_layer_output(model, layer_id, ratio)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        ratio = config.DEFAULT_NEURONS_SHRINK_RATIO
        actions: List[Action] = []
        for layer_id in get_all_hidden_modules(gm):
            mod = get_layer_module(layer_id, gm)
            if not isinstance(mod, nn.Linear):
                continue
            new_out = max(1, int(mod.out_features * ratio))
            if new_out >= mod.out_features or new_out < config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
                continue
            node = _find_call_module(gm.graph.nodes, layer_id)
            if propagation_hits_unsizable(gm, node):
                continue
            actions.append(DelNeurons([layer_id, ratio]))
        return actions

    def __str__(self):
        return " ( Delete Neurons Action: " + str(self.params) + " ) "
