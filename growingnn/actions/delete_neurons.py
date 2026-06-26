from typing import List

from torch import fx, nn

from growingnn.utils.fx import ModuleResolver, NodeEditor, NodeWidthAnalyser, GraphStructureQuery
from growingnn.actions.utils.layer_resize import propagate_neuron_change
from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.core import config
from .action import Action

def resize_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, new_width: int) -> fx.GraphModule:
    """Resize a Linear layer's output to new_width and propagate the change through the graph."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is {type(mod).__name__}, not nn.Linear")
    NodeEditor.replace_submodule(gm, layer_id, LinearFactory.create_linear_with_rescaled_neurons(mod, new_width))
    propagate_neuron_change(gm, ModuleResolver.find_call_module(gm.graph.nodes, layer_id), new_width, set())
    gm.recompile()
    for tensor in list(gm.parameters()) + list(gm.buffers()):
        if tensor.numel() > 0 and not tensor.is_contiguous():
            tensor.data = tensor.data.contiguous()


def shrink_layer_output(gm: nn.Module | fx.GraphModule, layer_id: str, ratio: float) -> fx.GraphModule:
    """Shrink a Linear layer's output by ratio and propagate shapes."""
    gm = gm if isinstance(gm, fx.GraphModule) else fx.symbolic_trace(gm)
    mod = ModuleResolver.get_layer_module(layer_id, gm)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{layer_id} is not nn.Linear")
    new = max(1, int(mod.out_features * ratio))
    if new >= mod.out_features or new < config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
        return gm
    node = ModuleResolver.find_call_module(gm.graph.nodes, layer_id)
    if NodeWidthAnalyser.propagation_hits_unsizable(gm, node):
        return gm
    resize_layer_output(gm, layer_id, new)


class DelNeurons(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        layer_id = self.params[0]
        ratio = self.params[1] if len(self.params) > 1 else config.DEFAULT_NEURONS_SHRINK_RATIO
        shrink_layer_output(model, layer_id, ratio)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        model: nn.Module | fx.GraphModule,
        ratio: float = config.DEFAULT_NEURONS_SHRINK_RATIO,
    ) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        actions: List[Action] = []
        for layer_id in GraphStructureQuery.get_all_hidden_modules(gm):
            mod = ModuleResolver.get_layer_module(layer_id, gm)
            if not isinstance(mod, nn.Linear):
                continue
            new_out = max(1, int(mod.out_features * ratio))
            if new_out >= mod.out_features or new_out < config.MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL:
                continue
            node = ModuleResolver.find_call_module(gm.graph.nodes, layer_id)
            if NodeWidthAnalyser.propagation_hits_unsizable(gm, node):
                continue
            actions.append(DelNeurons([layer_id, ratio]))
        return actions

    def __str__(self):
        return " ( Delete Neurons Action: " + str(self.params) + " ) "
