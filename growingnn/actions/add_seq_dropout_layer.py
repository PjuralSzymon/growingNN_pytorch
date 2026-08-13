from typing import List

from torch import nn

from growingnn.actions.utils.regularization_factory import RegularizationFactory
from growingnn.actions.utils.seq_insertion import iter_seq_shape_matched_pairs
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.core.logger import logger
from growingnn.utils.fx import ModuleResolver, ModelStructureEditor, GraphStructureQuery
from .action import Action


def _is_dropout_module(mod: nn.Module | None) -> bool:
    """Return True when the module is nn.Dropout or nn.Dropout2d."""
    return mod is not None and isinstance(mod, config.DROPOUT_TYPES)


class AddSeqDropoutLayer(Action):

    def _execute(self, traced: TracedModel):
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
        for cand in iter_seq_shape_matched_pairs(traced):
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
        return actions

    def __str__(self):
        return " ( Add Seq Dropout Layer Action: " + str(self.params) + " ) "
