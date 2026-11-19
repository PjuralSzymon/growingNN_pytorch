from typing import List

from torch import fx, nn

from growingnn.core.traced_model import TracedModel
from growingnn.core.logger import logger
from growingnn.utils.fx import (
    ModuleResolver,
    FunctionalLinearResizeAnalyzer,
    describe_linear_operations,
    resize_functional_linear_output,
)
from growingnn.actions.utils.layer_resize import can_resize_linear_output, resize_layer_output
from growingnn.core import config
from .action import Action


def _functional_linear_growth_rejection_reason(gm: fx.GraphModule, plan) -> str | None:
    """Apply the existing absolute/small-model and relative/large-model budgets."""
    source_matrix = sum(
        (source.in_features or 0) * plan.new_width for source in plan.sources
    )
    sink_matrix = plan.new_width * (plan.sink.out_features or 0)
    parameter_count = sum(parameter.numel() for parameter in gm.parameters())
    if parameter_count < config.MIN_MODEL_PARAMETER_COUNT_FOR_RELATIVE_NEURON_GROWTH:
        if max(source_matrix, sink_matrix) > config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE:
            return "functional projection matrices exceed the absolute resize budget"
    else:
        budget = max(
            config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE,
            int(parameter_count * config.MAX_MODEL_RELATIVE_NEURON_GROWTH_FRACTION),
        )
        if plan.added_parameter_count > budget:
            return (
                f"functional resize adds {plan.added_parameter_count} parameters, "
                f"exceeding model-relative budget {budget}"
            )
    return None


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
    def _execute(self, traced: TracedModel):
        layer_id = self.params[0]
        ratio = self.params[1] if len(self.params) > 1 else config.DEFAULT_NEURONS_GROW_RATIO
        if len(self.params) > 2 and self.params[2] == "fx_functional":
            descriptor = next(
                (
                    item for item in describe_linear_operations(traced.gm)
                    if item.node.name == layer_id
                ),
                None,
            )
            if descriptor is None or descriptor.out_features is None:
                raise ValueError(f"Functional Linear target {layer_id!r} is missing from the graph")
            new_width = max(1, int(descriptor.out_features * ratio))
            plan, reason = FunctionalLinearResizeAnalyzer(traced.gm).plan(layer_id, new_width)
            if plan is None:
                raise ValueError(reason or "Functional Linear resize is unsupported")
            budget_reason = _functional_linear_growth_rejection_reason(traced.gm, plan)
            if budget_reason is not None:
                raise ValueError(budget_reason)
            resize_functional_linear_output(traced.gm, layer_id, new_width)
            traced.invalidate()
            return
        expand_layer_output(traced.gm, layer_id, ratio)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        ratio: float = config.DEFAULT_NEURONS_GROW_RATIO,
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
            actions.append(AddNeurons([layer_id, ratio]))

        functional_descriptors = [
            item for item in describe_linear_operations(gm)
            if item.form in ("function", "addmm")
        ]
        if functional_descriptors:
            try:
                traced.shapes()
            except Exception as exc:
                logger.info(
                    "AddNeurons skipped functional projections: shape/interface propagation failed: %s",
                    exc,
                )
                return actions
            # Shape propagation may add metadata required by structural resize
            # planning, so refresh operation descriptors against the annotated graph.
            analyzer = FunctionalLinearResizeAnalyzer(gm)
            functional_descriptors = [
                item for item in analyzer.descriptors
                if item.form in ("function", "addmm")
            ]
        else:
            analyzer = FunctionalLinearResizeAnalyzer(gm)
        seen_functional_groups: set[tuple[str, ...]] = set()
        for descriptor in functional_descriptors:
            if descriptor.out_features is None:
                logger.debug(
                    "AddNeurons skip functional projection %s: output width is unknown",
                    descriptor.node.name,
                )
                continue
            new_width = max(1, int(descriptor.out_features * ratio))
            plan, reason = analyzer.plan(descriptor.node, new_width)
            if plan is None:
                logger.debug(
                    "AddNeurons skip functional projection %s: %s",
                    descriptor.node.name, reason,
                )
                continue
            budget_reason = _functional_linear_growth_rejection_reason(gm, plan)
            if budget_reason is not None:
                logger.debug(
                    "AddNeurons skip functional projection %s: %s",
                    descriptor.node.name, budget_reason,
                )
                continue
            group_key = tuple(sorted(
                [source.node.name for source in plan.sources] + [plan.sink.node.name]
            ))
            if group_key in seen_functional_groups:
                continue
            seen_functional_groups.add(group_key)
            actions.append(AddNeurons([descriptor.node.name, ratio, "fx_functional"]))
        return actions

    def __str__(self):
        return " ( Add Neurons Action: " + str(self.params) + " ) "
