"""Aggregate architecture mutation actions for a traced model."""

from __future__ import annotations

from collections.abc import Iterable

import torch.fx as fx
import torch.nn as nn

from growingnn.actions.action import Action, Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_layer import AddSeqLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.core.config import RunningConfig


def generate_all_actions(
    model: nn.Module | fx.GraphModule, config: RunningConfig) -> list[Action]:
    actions: list[Action] = []
    if config.ACTIONS_ENABLE_ADD_RES_LAYER:
        actions.extend(AddResLayer.generate_all_actions(model, layer_types=(Layer_Type.EYE, Layer_Type.ZERO)))
    if config.ACTIONS_ENABLE_ADD_RES_CONV_LAYER:
        actions.extend(AddResConvLayer.generate_all_actions(model))
    if config.ACTIONS_ENABLE_ADD_SEQ_LAYER:
        actions.extend(AddSeqLayer.generate_all_actions(model))
    if config.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER:
        actions.extend(AddSeqConvLayer.generate_all_actions(model))
    if config.ACTIONS_ENABLE_DEL_LAYER:
        actions.extend(DelLayer.generate_all_actions(model))
    if config.ACTIONS_ENABLE_DEL_NEURONS_01:
        actions.extend(DelNeurons.generate_all_actions(model, ratio=0.1))
    if config.ACTIONS_ENABLE_DEL_NEURONS_05:
        actions.extend(DelNeurons.generate_all_actions(model, ratio=0.5))
    if config.ACTIONS_ENABLE_DEL_NEURONS_09:
        actions.extend(DelNeurons.generate_all_actions(model, ratio=0.9))
    return actions
