from collections import deque
from collections.abc import Mapping
from typing import List

from torch import fx, nn

from growingnn.actions.utils.layer_Factory import LinearFactory
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import (
    LayerBridgeFinder,
    ModuleResolver, ModelStructureEditor,
    LowRankFeatureAdapter, supported_adapter_ranks,
    analyze_feature_tensor, pair_traversal_decision,
)
from growingnn.core.logger import logger
from .action import Action, Layer_Type


class AddSeqLinearLayer(Action):

    @staticmethod
    def _path_edges(dst: fx.Node, src: fx.Node) -> set[tuple[fx.Node, fx.Node]]:
        """Find one backward path in O(V+E), avoiding recursive path enumeration."""
        queue = deque([dst])
        next_toward_dst: dict[fx.Node, fx.Node | None] = {dst: None}
        while queue:
            node = queue.popleft()
            if node is src:
                path = [src]
                while path[-1] is not dst:
                    path.append(next_toward_dst[path[-1]])
                return {
                    (path[index], path[index + 1])
                    for index in range(len(path) - 1)
                }
            for predecessor in node.all_input_nodes:
                if predecessor not in next_toward_dst:
                    next_toward_dst[predecessor] = node
                    queue.append(predecessor)
        return set()

    def _execute(self, traced: TracedModel):
        if len(self.params) > 4 and self.params[4] == "fx_edge":
            source_name, destination_name, layer, name = self.params[:4]
            nodes = {node.name: node for node in traced.gm.graph.nodes}
            source = nodes.get(source_name)
            destination = nodes.get(destination_name)
            if source is None or destination is None:
                raise ValueError(
                    f"FX edge action refers to missing nodes {source_name!r} -> {destination_name!r}"
                )
            ModelStructureEditor.insert_module_on_edge(
                traced.gm, source, destination, layer, name,
                use_path=self.params[5] if len(self.params) > 5 else None,
            )
            return
        ModelStructureEditor.add_new_seq_layer(traced.gm, self.params[0], self.params[1], self.params[2], self.params[3])

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        *,
        feature_axis: int | Mapping[int, int] = -1,
        supported_ranks: tuple[int, ...] = (2, 3),
    ) -> List[Action]:
        gm = traced.gm
        out_shapes, in_shapes = traced.shapes()
        actions: List[Action] = []
        pair_decision = pair_traversal_decision(gm)
        legacy_pairs = traced.sequential_pairs() if pair_decision.use_legacy_pairs else ()
        if not pair_decision.use_legacy_pairs:
            logger.info(
                "AddSeqLinearLayer using FX region edges instead of exhaustive module pairs: %s",
                "; ".join(pair_decision.reasons),
            )
        for layer_from_id, layer_to_id in legacy_pairs:
            s_out = out_shapes.get(layer_from_id)
            s_in = in_shapes.get(layer_to_id)
            sizes = LayerBridgeFinder.find_bridge_linear_sizes(s_out, s_in)
            if sizes is not None:
                if sizes[0] * sizes[1] > config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE:
                    continue
                name = ModuleResolver.unique_call_module_name("seq_linear", gm)
                layer = LinearFactory.create_linear(sizes[0], sizes[1], Layer_Type.EYE)
                logger.debug(
                    "AddSeqLinearLayer %s -> %s: Linear(%d, %d) out=%s in=%s",
                    layer_from_id,
                    layer_to_id,
                    sizes[0],
                    sizes[1],
                    s_out,
                    s_in,
                )
                actions.append(AddSeqLinearLayer([layer_from_id, layer_to_id, layer, name]))
                continue
            conv_linear_sizes = LayerBridgeFinder.find_seq_linear_after_conv_sizes(s_out, s_in)
            if conv_linear_sizes is None:
                logger.debug("AddSeqLinearLayer skip %s -> %s", layer_from_id, layer_to_id)
                continue
            if conv_linear_sizes[0] * conv_linear_sizes[1] > config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE:
                continue
            name = ModuleResolver.unique_call_module_name("seq_linear", gm)
            layer = LinearFactory.create_linear(
                conv_linear_sizes[0],
                conv_linear_sizes[1],
                Layer_Type.EYE,
            )
            logger.debug(
                "AddSeqLinearLayer %s -> %s: conv->linear (%d, %d) out=%s in=%s",
                layer_from_id,
                layer_to_id,
                conv_linear_sizes[0],
                conv_linear_sizes[1],
                s_out,
                s_in,
            )
            actions.append(AddSeqLinearLayer([layer_from_id, layer_to_id, layer, name]))

        # Keep the legacy module-pair proposals above for existing callers and
        # ResNet behavior. Also expose graph-edge proposals at architectural seams
        # when the producer is functional, a merge/split, or a module boundary.
        # This makes rank-3 Transformer activations and non-module FX operations
        # usable without proposing an edit at every node in the graph.
        legacy_covered_edges: set[tuple[fx.Node, fx.Node]] = set()
        module_nodes = [node for node in gm.graph.nodes if node.op == "call_module"]
        for legacy_action in actions:
            source_target, destination_target = legacy_action.params[:2]
            source_node = next((n for n in module_nodes if str(n.target) == source_target), None)
            destination_node = next((n for n in module_nodes if str(n.target) == destination_target), None)
            if source_node is None or destination_node is None:
                continue
            legacy_covered_edges.update(
                AddSeqLinearLayer._path_edges(destination_node, source_node)
            )

        observed_edges: set[tuple[fx.Node, fx.Node]] = set()
        for site in traced.action_sites(include_rejected=False):
            source, destination = site.edge.source, site.edge.destination
            if source is None or destination is None or (source, destination) in observed_edges:
                continue
            observed_edges.add((source, destination))
            if (source, destination) in legacy_covered_edges:
                continue
            if not site.meaningful_seam:
                continue
            feature = analyze_feature_tensor(
                site.interface,
                feature_axis=feature_axis,
                supported_ranks=supported_ranks,
            )
            if not feature.eligible:
                logger.debug(
                    "AddSeqLinearLayer FX skip %s -> %s: %s",
                    source.name, destination.name, feature.rejection_reason,
                )
                continue
            width = feature.features
            dtype = feature.dtype
            device = feature.device or "cpu"
            for rank in supported_adapter_ranks(
                width, config.MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE
            ):
                layer = LowRankFeatureAdapter(
                    width,
                    rank,
                    residual=True,
                    feature_axis=feature.feature_axis,
                ).to(
                    device=device, dtype=dtype
                )
                name = ModuleResolver.unique_call_module_name("seq_linear", gm)
                use_paths = site.edge.use_paths or (None,)
                for use_path in use_paths:
                    action_name = name if len(use_paths) == 1 else ModuleResolver.unique_call_module_name("seq_linear", gm)
                    action_layer = layer if len(use_paths) == 1 else LowRankFeatureAdapter(
                        width, rank, residual=True, feature_axis=feature.feature_axis
                    ).to(device=device, dtype=dtype)
                    params = [source.name, destination.name, action_layer, action_name, "fx_edge"]
                    if use_path is not None:
                        params.append(use_path)
                    actions.append(AddSeqLinearLayer(params))
        return actions

    def __str__(self):
        return " ( Add Seq Linear Layer Action: " + str(self.params) + " ) "
