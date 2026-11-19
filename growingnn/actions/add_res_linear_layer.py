from typing import Iterable, List
from collections.abc import Mapping
from collections import deque

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    LayerBridgeFinder,
    ModuleResolver, ModelStructureEditor,
    LowRankFeatureAdapter, supported_adapter_ranks,
    GraphRegionAnalyzer, analyze_feature_tensor,
    pair_traversal_decision,
)
from growingnn.utils.fx.sum_nodes import is_sum_node
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddResLinearLayer(Action):

    @staticmethod
    def _module_pair_is_local(gm: fx.GraphModule, source_target: str, destination_target: str) -> bool:
        """Accept legacy residual pairs only when no other module lies between them."""
        sources = [
            node for node in gm.graph.nodes
            if node.op == "call_module" and str(node.target) == source_target
        ]
        destinations = {
            node for node in gm.graph.nodes
            if node.op == "call_module" and str(node.target) == destination_target
        }
        queue = deque(sources)
        seen = set(sources)
        while queue:
            node = queue.popleft()
            if node in destinations:
                return True
            for user in node.users:
                if user in seen:
                    continue
                if user.op == "call_module":
                    if user in destinations:
                        return True
                    continue
                seen.add(user)
                queue.append(user)
        return False

    def _execute(self, traced: TracedModel):
        if len(self.params) > 4 and self.params[4] == "fx_residual":
            source_name, destination_name, layer, name = self.params[:4]
            nodes = {node.name: node for node in traced.gm.graph.nodes}
            source = nodes.get(source_name)
            destination = nodes.get(destination_name)
            if source is None or destination is None:
                raise ValueError(
                    f"FX residual action refers to missing nodes {source_name!r} -> {destination_name!r}"
                )
            ModelStructureEditor.add_residual_branch_on_nodes(
                traced.gm, destination, source, layer, name
            )
            return
        ModelStructureEditor.add_new_residual_layer(traced.gm, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        layer_types: Iterable[Layer_Type] = Layer_Type,
        *,
        feature_axis: int | Mapping[int, int] = -1,
        supported_ranks: tuple[int, ...] = (2, 3),
    ) -> List[Action]:
        gm = traced.gm
        layer_types = tuple(layer_types)
        out_shapes, _ = traced.shapes()
        actions: List[Action] = []
        pair_decision = pair_traversal_decision(gm)
        legacy_pairs = traced.dependency_pairs() if pair_decision.use_legacy_pairs else ()
        if not pair_decision.use_legacy_pairs:
            logger.info(
                "AddResLinearLayer using FX merge sites instead of exhaustive module pairs: %s",
                "; ".join(pair_decision.reasons),
            )
        for layer_from_id, layer_to_id in legacy_pairs:
            if not AddResLinearLayer._module_pair_is_local(
                gm, layer_from_id, layer_to_id
            ):
                logger.debug(
                    "AddResLinearLayer skip %s -> %s: another module lies between endpoints",
                    layer_from_id, layer_to_id,
                )
                continue
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

        # Generate residual growth from actual FX merge nodes as well as legacy
        # module pairs. Each branch starts from one operand already consumed by
        # the merge, so discovery is tied to graph structure rather than distant
        # same-shape module pairs.
        for merge in gm.graph.nodes:
            if not is_sum_node(merge) or merge.meta.get("growingnn_generated_merge"):
                continue
            merge_interface = GraphRegionAnalyzer.describe_node_value(merge)
            merge_feature = analyze_feature_tensor(
                merge_interface,
                feature_axis=feature_axis,
                supported_ranks=supported_ranks,
            )
            if not merge_feature.eligible:
                logger.debug(
                    "AddResLinearLayer FX skip merge %s: %s",
                    merge.name, merge_feature.rejection_reason,
                )
                continue
            width = merge_feature.features
            for source in merge.all_input_nodes:
                source_interface = GraphRegionAnalyzer.describe_node_value(source)
                source_feature = analyze_feature_tensor(
                    source_interface,
                    feature_axis=feature_axis,
                    supported_ranks=supported_ranks,
                )
                if not source_feature.eligible:
                    logger.debug(
                        "AddResLinearLayer FX skip %s -> %s: %s",
                        source.name, merge.name, source_feature.rejection_reason,
                    )
                    continue
                if source_interface.shape != merge_interface.shape:
                    logger.debug(
                        "AddResLinearLayer FX skip %s -> %s: boundary shapes differ (%s vs %s)",
                        source.name, merge.name, source_interface.shape, merge_interface.shape,
                    )
                    continue
                if source_feature.dtype != merge_feature.dtype:
                    logger.debug(
                        "AddResLinearLayer FX skip %s -> %s: boundary dtypes differ (%s vs %s)",
                        source.name, merge.name, source_feature.dtype, merge_feature.dtype,
                    )
                    continue
                if source_feature.device != merge_feature.device:
                    logger.debug(
                        "AddResLinearLayer FX skip %s -> %s: boundary devices differ (%s vs %s)",
                        source.name, merge.name, source_feature.device, merge_feature.device,
                    )
                    continue
                dtype = source_feature.dtype
                device = source_feature.device or "cpu"
                if Layer_Type.ZERO not in layer_types:
                    continue
                for rank in supported_adapter_ranks(
                    width, config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE
                ):
                    name = ModuleResolver.unique_call_module_name(
                        "res_adapter", gm
                    )
                    layer = LowRankFeatureAdapter(
                        width,
                        rank,
                        residual=False,
                        feature_axis=merge_feature.feature_axis,
                    ).to(
                        device=device, dtype=dtype
                    )
                    actions.append(
                        AddResLinearLayer(
                            [source.name, merge.name, layer, name, "fx_residual", rank]
                        )
                    )
        return actions

    def __str__(self):
        return " ( Add Res Linear Layer Action: " + str(self.params) + " ) "
