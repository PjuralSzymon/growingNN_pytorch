from typing import List

from torch import fx, nn

from growingnn.utils.fx import (
    LayerBridgeFinder,
    LayerShapeAnalyser,
    GraphStructureQuery,
    ModelStructureEditor,
)
from growingnn.utils.fx.graph_editor import (
    branch_only_bypass_compatible,
    compute_bypass_matching,
)
from growingnn.utils.fx.sum_nodes import is_merge_branch_layer
from .action import Action


def _find_layer_node(gm: fx.GraphModule, layer_id: str) -> fx.Node:
    return next(
        n for n in gm.graph.nodes
        if n.op == "call_module" and n.target == layer_id
    )


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


def can_bypass_delete_layer(
    model: nn.Module | fx.GraphModule,
    layer_id: str,
    output_shapes: dict[str, tuple[int, ...]] | None = None,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> bool:
    """Return True when each successor can be fed by one shape-compatible predecessor."""
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    if output_shapes is None or input_shapes is None:
        output_shapes, input_shapes = LayerShapeAnalyser.collect_layer_shapes(gm)
    layer_node = _find_layer_node(gm, layer_id)
    if is_merge_branch_layer(layer_node):
        return bool(layer_node.all_input_nodes)
    input_layers = GraphStructureQuery.get_input_layers(layer_id, gm)
    output_layers = GraphStructureQuery.get_output_layers(layer_id, gm)
    if not input_layers:
        return False
    if not output_layers:
        return branch_only_bypass_compatible(layer_node, input_shapes)
    return compute_bypass_matching(input_layers, output_layers, output_shapes, input_shapes) is not None


class DelLayer(Action):
    def execute(self, model: nn.Module | fx.GraphModule):
        ModelStructureEditor.delete_layer(model, self.params[0])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(model: nn.Module | fx.GraphModule) -> List[Action]:
        gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        output_shapes, input_shapes = LayerShapeAnalyser.collect_layer_shapes(gm)
        actions: List[Action] = []
        for layer_id in dict.fromkeys(GraphStructureQuery.get_all_hidden_modules(gm)):
            if can_bypass_delete_layer(gm, layer_id, output_shapes, input_shapes):
                actions.append(DelLayer([layer_id]))
        return actions

    def __str__(self):
        return " ( Delete Layer Action: " + str(self.params) + " ) "


def explain_delete_layer_blockers(
    gm: fx.GraphModule,
) -> list[tuple[str, str]]:
    """Return (layer_id, reason) for hidden modules that cannot be deleted."""
    output_shapes, input_shapes = LayerShapeAnalyser.collect_layer_shapes(gm)
    blockers: list[tuple[str, str]] = []
    for layer_id in dict.fromkeys(GraphStructureQuery.get_all_hidden_modules(gm)):
        if can_bypass_delete_layer(gm, layer_id, output_shapes, input_shapes):
            continue
        layer_node = _find_layer_node(gm, layer_id)
        if is_merge_branch_layer(layer_node):
            blockers.append((layer_id, "merge branch without FX input"))
            continue
        input_layers = GraphStructureQuery.get_input_layers(layer_id, gm)
        output_layers = GraphStructureQuery.get_output_layers(layer_id, gm)
        if not input_layers:
            blockers.append((layer_id, "no editable predecessors"))
            continue
        if not output_layers:
            if not branch_only_bypass_compatible(layer_node, input_shapes):
                blockers.append((layer_id, "no shape-compatible branch-only bypass"))
            continue
        matching = compute_bypass_matching(input_layers, output_layers, output_shapes, input_shapes)
        blockers.append((
            layer_id,
            f"no shape-compatible matching in={input_layers} out={output_layers} got={matching}",
        ))
    return blockers
