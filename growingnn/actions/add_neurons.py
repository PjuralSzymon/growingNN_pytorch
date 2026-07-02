from typing import List

from torch import fx, nn

from growingnn.utils.fx import ModuleResolver, GraphStructureQuery
from growingnn.actions.utils.layer_resize import can_resize_linear_output, resize_layer_output
from growingnn.core import config
from .action import Action


def expand_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, ratio: float) -> fx.GraphModule:
    """Grow a Linear layer's output by ratio and propagate shapes."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is not nn.Linear")
    new = max(1, int(mod.out_features * ratio))
    if not can_resize_linear_output(gm, layer_id, new):
        return gm
    return resize_layer_output(gm, layer_id, new)


class AddNeurons(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        layer_id = self.params[0]
        ratio = self.params[1] if len(self.params) > 1 else config.DEFAULT_NEURONS_GROW_RATIO
        expand_layer_output(model, layer_id, ratio)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        model: nn.Module | fx.GraphModule,
        ratio: float = config.DEFAULT_NEURONS_GROW_RATIO,
    ) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        actions: List[Action] = []
        for layer_id in GraphStructureQuery.get_all_hidden_modules(gm):
            mod = ModuleResolver.get_layer_module(layer_id, gm)
            if not isinstance(mod, nn.Linear):
                continue
            new_out = max(1, int(mod.out_features * ratio))
            if not can_resize_linear_output(gm, layer_id, new_out):
                continue
            actions.append(AddNeurons([layer_id, ratio]))
        return actions

    def __str__(self):
        return " ( Add Neurons Action: " + str(self.params) + " ) "
