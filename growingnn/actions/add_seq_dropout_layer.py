from typing import List

from torch import nn

from growingnn.actions.utils.regularization_factory import RegularizationFactory
from growingnn.actions.utils.seq_insertion import iter_seq_shape_matched_pairs
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.core.logger import logger
from growingnn.utils.fx import (
    ModuleResolver, ModelStructureEditor, GraphStructureQuery, pair_traversal_decision,
)
from torch import fx
from .action import Action


def _is_dropout_module(mod: nn.Module | None) -> bool:
    """Return True when the module is nn.Dropout or nn.Dropout2d."""
    return mod is not None and isinstance(mod, config.DROPOUT_TYPES)


class AddSeqDropoutLayer(Action):

    def _execute(self, traced: TracedModel):
        if len(self.params) > 4 and self.params[4] == "fx_edge":
            source_name, destination_name, layer, name = self.params[:4]
            nodes = {node.name: node for node in traced.gm.graph.nodes}
            source = nodes.get(source_name)
            destination = nodes.get(destination_name)
            if source is None or destination is None:
                raise ValueError(
                    f"FX dropout action refers to missing nodes {source_name!r} -> {destination_name!r}"
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
        p: float = config.DEFAULT_DROPOUT_RATE,
    ) -> List[Action]:
        gm = traced.gm
        actions: List[Action] = []
        nodes = list(gm.graph.nodes)
        pair_decision = pair_traversal_decision(gm)
        use_fx_first = not pair_decision.use_legacy_pairs
        if not use_fx_first:
            legacy_candidates = iter_seq_shape_matched_pairs(traced)
        else:
            legacy_candidates = ()
            logger.debug(
                "AddSeqDropoutLayer using FX edge sites instead of exhaustive module pairs: %s",
                "; ".join(pair_decision.reasons),
            )
        for cand in legacy_candidates:
            from_mod = ModuleResolver.get_layer_module(cand.from_id, gm)
            to_mod = ModuleResolver.get_layer_module(cand.to_id, gm)
            if _is_dropout_module(from_mod) or _is_dropout_module(to_mod):
                logger.debug("AddSeqDropoutLayer skip %s -> %s: adjacent to dropout", cand.from_id, cand.to_id)
                continue
            if GraphStructureQuery.path_has_dropout(gm, cand.from_id, cand.to_id):
                logger.debug("AddSeqDropoutLayer skip %s -> %s: dropout already on path", cand.from_id, cand.to_id)
                continue
            layer = RegularizationFactory.create_dropout(cand.shape, p)
            if layer is None:
                logger.debug("AddSeqDropoutLayer skip %s -> %s: rank %d", cand.from_id, cand.to_id, len(cand.shape))
                continue
            name = ModuleResolver.unique_call_module_name("seq_dropout", gm)
            logger.debug("AddSeqDropoutLayer %s -> %s: p=%s shape=%s", cand.from_id, cand.to_id, p, cand.shape)
            actions.append(AddSeqDropoutLayer([cand.from_id, cand.to_id, layer, name]))

        if use_fx_first:
            traced.shapes()
            observed: set[tuple[fx.Node, fx.Node]] = set()
            for site in traced.action_sites(include_rejected=False):
                source, destination = site.edge.source, site.edge.destination
                if source is None or destination is None or (source, destination) in observed:
                    continue
                observed.add((source, destination))
                if not site.meaningful_seam:
                    continue
                if any(
                    node.op == "call_module"
                    and isinstance(ModuleResolver.get_layer_module(node, gm), config.DROPOUT_TYPES)
                    for node in (source, destination)
                ):
                    continue
                name = ModuleResolver.unique_call_module_name("seq_dropout", gm)
                use_paths = site.edge.use_paths or (None,)
                for use_path in use_paths:
                    action_name = name if len(use_paths) == 1 else ModuleResolver.unique_call_module_name("seq_dropout", gm)
                    params = [source.name, destination.name, nn.Dropout(p=p), action_name, "fx_edge"]
                    if use_path is not None:
                        params.append(use_path)
                    actions.append(AddSeqDropoutLayer(params))
        return actions

    def __str__(self):
        return " ( Add Seq Dropout Layer Action: " + str(self.params) + " ) "
