from typing import Iterable, List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    LayerBridgeFinder,
    ModuleResolver, ModelStructureEditor,
)
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddResLinearLayer(Action):

    def _execute(self, traced: TracedModel):
        ModelStructureEditor.add_new_residual_layer(traced.gm, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        layer_types: Iterable[Layer_Type] = Layer_Type,
    ) -> List[Action]:
        gm = traced.gm
        out_shapes, _ = traced.shapes()
        actions: List[Action] = []
        for layer_from_id, layer_to_id in traced.dependency_pairs():
            sizes = LayerBridgeFinder.find_bridge_res_linear_sizes(
                out_shapes.get(layer_from_id),
                out_shapes.get(layer_to_id),
            )
            if sizes is None:
                logger.debug("AddResLinearLayer skip %s -> %s", layer_from_id, layer_to_id)
                continue
            if sizes[0] * sizes[1] > config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE:
                continue
            for layer_type in layer_types:
                name = ModuleResolver.unique_call_module_name(f"res_linear_{layer_type.name}", gm)
                layer = LinearFactory.create_linear(sizes[0], sizes[1], layer_type)
                logger.debug("AddResLinearLayer %s -> %s: Linear(%d, %d) %s out=%s/%s", layer_from_id, layer_to_id, sizes[0], sizes[1], layer_type.name, out_shapes.get(layer_from_id), out_shapes.get(layer_to_id))
                actions.append(AddResLinearLayer([layer_from_id, layer_to_id, layer, name]))
        return actions

    def __str__(self):
        return " ( Add Res Linear Layer Action: " + str(self.params) + " ) "
