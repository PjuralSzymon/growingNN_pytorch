from typing import List

from torch import fx, nn

from growingnn.utils.fx import ModuleResolver, NodeWidthAnalyser, GraphStructureQuery
from growingnn.actions.delete_neurons import resize_layer_output
from growingnn.core import config
from .action import Action


def _grow_within_matrix_limit(mod: nn.Linear, new_out: int) -> bool:
    """Return False when grow would need an oversized rescale matrix or weight tensor."""
    max_side = max(mod.out_features, new_out)
    return (
        max_side * max_side <= config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE
        and mod.in_features * new_out <= config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE
    )


def expand_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, ratio: float) -> fx.GraphModule:
    """Grow a Linear layer's output by ratio and propagate shapes."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is not nn.Linear")
    new = max(1, int(mod.out_features * ratio))
    if new <= mod.out_features:
        return gm
    if not _grow_within_matrix_limit(mod, new):
        return gm
    node = ModuleResolver.find_call_module(gm.graph.nodes, layer_id)
    if NodeWidthAnalyser.propagation_hits_unsizable(gm, node):
        return gm
    resize_layer_output(gm, layer_id, new)


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
            if new_out <= mod.out_features:
                continue
            if not _grow_within_matrix_limit(mod, new_out):
                continue
            node = ModuleResolver.find_call_module(gm.graph.nodes, layer_id)
            if NodeWidthAnalyser.propagation_hits_unsizable(gm, node):
                continue
            actions.append(AddNeurons([layer_id, ratio]))
        return actions

    def __str__(self):
        return " ( Add Neurons Action: " + str(self.params) + " ) "
