from typing import List

from torch import fx, nn

from growingnn.utils.fx import (
    LayerBridgeFinder, LayerShapeAnalyser,
    GraphStructureQuery, ModelStructureEditor,
)
from .action import Action


def _shapes_for_layers(
    shape_map: dict[str, tuple[int, ...]],
    layer_ids: list[str],
) -> list[tuple[int, ...] | None]:
    return [shape_map.get(layer_id) for layer_id in layer_ids]


def has_same_output_shape(
    model: nn.Module | fx.GraphModule,
    input_layers: list[str],
    output_shapes: dict[str, tuple[int, ...]] | None = None,
) -> bool:
    if not input_layers:
        return False
    if output_shapes is None:
        if not isinstance(model, fx.GraphModule):
            return False
        output_shapes = LayerShapeAnalyser.get_layer_output_shapes(model)
    return (
        LayerBridgeFinder.uniform_activation_shape(_shapes_for_layers(output_shapes, input_layers))
        is not None
    )


def has_same_input_shape(
    model: nn.Module | fx.GraphModule,
    output_layers: list[str],
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> bool:
    if not output_layers:
        return False
    if input_shapes is None:
        if not isinstance(model, fx.GraphModule):
            return False
        input_shapes = LayerShapeAnalyser.get_layer_input_shapes(model)
    return (
        LayerBridgeFinder.uniform_activation_shape(_shapes_for_layers(input_shapes, output_layers))
        is not None
    )


def get_common_output_shape(
    model: nn.Module | fx.GraphModule,
    input_layers: list[str],
    output_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[int, ...] | None:
    if output_shapes is None:
        if not isinstance(model, fx.GraphModule):
            return None
        output_shapes = LayerShapeAnalyser.get_layer_output_shapes(model)
    return LayerBridgeFinder.uniform_activation_shape(_shapes_for_layers(output_shapes, input_layers))


def get_common_input_shape(
    model: nn.Module | fx.GraphModule,
    output_layers: list[str],
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[int, ...] | None:
    if input_shapes is None:
        if not isinstance(model, fx.GraphModule):
            return None
        input_shapes = LayerShapeAnalyser.get_layer_input_shapes(model)
    return LayerBridgeFinder.uniform_activation_shape(_shapes_for_layers(input_shapes, output_layers))


class DelLayer(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        ModelStructureEditor.delete_layer(model, self.params[0])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        output_shapes = LayerShapeAnalyser.get_layer_output_shapes(gm)
        input_shapes = LayerShapeAnalyser.get_layer_input_shapes(gm)
        actions: List[Action] = []
        for layer_id in GraphStructureQuery.get_all_hidden_modules(gm):
            input_layers = GraphStructureQuery.get_input_layers(layer_id, gm)
            output_layers = GraphStructureQuery.get_output_layers(layer_id, gm)
            in_shape = get_common_output_shape(gm, input_layers, output_shapes)
            out_shape = get_common_input_shape(gm, output_layers, input_shapes)
            if in_shape is None or out_shape is None or in_shape != out_shape:
                continue
            actions.append(DelLayer([layer_id]))
        return actions

    def __str__(self):
        return " ( Delete Layer Action: " + str(self.params) + " ) "
