from typing import List

from torch import fx, nn

from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import ModuleResolver
from growingnn.utils.fx.graph_extraction import extract_graph
from growingnn.actions.utils.layer_resize import can_resize_linear_output, resize_layer_output
from growingnn.core import config
from .action import Action


def shrink_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, ratio: float) -> fx.GraphModule:
    """Shrink a Linear layer's output by ratio and propagate shapes."""
    gm = extract_graph(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is not nn.Linear")
    new = max(1, int(mod.out_features * ratio))
    if not can_resize_linear_output(gm, layer_id, new):
        return gm
    return resize_layer_output(gm, layer_id, new)


class DelNeurons(Action):
    def _execute(self, traced: TracedModel):
        layer_id = self.params[0]
        ratio = self.params[1] if len(self.params) > 1 else config.DEFAULT_NEURONS_SHRINK_RATIO
        shrink_layer_output(traced.gm, layer_id, ratio)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        ratio: float = config.DEFAULT_NEURONS_SHRINK_RATIO,
    ) -> List[Action]:
        gm = traced.gm
        actions: List[Action] = []
        for layer_id in traced.hidden_modules():
            mod = ModuleResolver.get_layer_module(layer_id, gm)
            if not isinstance(mod, nn.Linear):
                continue
            new_out = max(1, int(mod.out_features * ratio))
            if not can_resize_linear_output(gm, layer_id, new_out):
                continue
            actions.append(DelNeurons([layer_id, ratio]))
        return actions

    def __str__(self):
        return " ( Delete Neurons Action: " + str(self.params) + " ) "
