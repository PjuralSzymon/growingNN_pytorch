"""Aggregate architecture mutation actions for a traced model."""

from __future__ import annotations

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.add_neurons import AddNeurons
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel


def generate_all_actions(traced: TracedModel, config: RunningConfig) -> list[Action]:
    actions: list[Action] = []
    if config.ACTIONS_ENABLE_ADD_RES_LAYER:
        actions.extend(AddResLinearLayer.generate_all_actions(traced, layer_types=(Layer_Type.EYE, Layer_Type.ZERO)))
    if config.ACTIONS_ENABLE_ADD_RES_CONV_LAYER:
        actions.extend(AddResConvLayer.generate_all_actions(traced))
    if config.ACTIONS_ENABLE_ADD_SEQ_LAYER:
        actions.extend(AddSeqLinearLayer.generate_all_actions(traced))
    if config.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER:
        actions.extend(AddSeqConvLayer.generate_all_actions(traced))
    if config.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01:
        actions.extend(AddSeqDropoutLayer.generate_all_actions(traced, p=0.1))
    if config.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_02:
        actions.extend(AddSeqDropoutLayer.generate_all_actions(traced, p=0.2))
    if config.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05:
        actions.extend(AddSeqDropoutLayer.generate_all_actions(traced, p=0.5))
    if config.ACTIONS_ENABLE_ADD_NEURONS_11:
        actions.extend(AddNeurons.generate_all_actions(traced, ratio=1.1))
    if config.ACTIONS_ENABLE_ADD_NEURONS_15:
        actions.extend(AddNeurons.generate_all_actions(traced, ratio=1.5))
    if config.ACTIONS_ENABLE_ADD_NEURONS_20:
        actions.extend(AddNeurons.generate_all_actions(traced, ratio=2.0))
    if config.ACTIONS_ENABLE_DEL_LAYER:
        actions.extend(DelLayer.generate_all_actions(traced))
    if config.ACTIONS_ENABLE_DEL_NEURONS_01:
        actions.extend(DelNeurons.generate_all_actions(traced, ratio=0.1))
    if config.ACTIONS_ENABLE_DEL_NEURONS_05:
        actions.extend(DelNeurons.generate_all_actions(traced, ratio=0.5))
    if config.ACTIONS_ENABLE_DEL_NEURONS_09:
        actions.extend(DelNeurons.generate_all_actions(traced, ratio=0.9))
    return actions
