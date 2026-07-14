from typing import List

from torch import fx, nn

from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
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


def can_bypass_delete_layer(
    model: nn.Module | fx.GraphModule,
    layer_id: str,
    output_shapes: dict[str, tuple[int, ...]] | None = None,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    input_shape: tuple[int, ...] | None = None,
) -> bool:
    """Return True when each successor can be fed by one shape-compatible predecessor."""
    gm = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    if output_shapes is None or input_shapes is None:
        output_shapes, input_shapes = LayerShapeAnalyser.collect_layer_shapes(
            gm, input_shape=input_shape
        )
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
    def _execute(self, traced: TracedModel):
        ModelStructureEditor.delete_layer(
            traced.gm, self.params[0], input_shape=traced.input_shape
        )

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(traced: TracedModel) -> List[Action]:
        gm = traced.gm
        output_shapes, input_shapes = traced.shapes()
        actions: List[Action] = []
        for layer_id in dict.fromkeys(traced.hidden_modules()):
            if can_bypass_delete_layer(gm, layer_id, output_shapes, input_shapes):
                actions.append(DelLayer([layer_id]))
        return actions

    def __str__(self):
        return " ( Delete Layer Action: " + str(self.params) + " ) "


def explain_delete_layer_blockers(
    graph: TracedModel,
) -> list[tuple[str, str]]:
    """Return (layer_id, reason) for hidden modules that cannot be deleted."""
    gm = graph.gm
    output_shapes, input_shapes = graph.shapes()
    blockers: list[tuple[str, str]] = []
    for layer_id in dict.fromkeys(graph.hidden_modules()):
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
