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


def generate_all_actions(
    model: nn.Module | fx.GraphModule,
    *,
    grow: bool = True,
    shrink: bool = True,
    layer_types: Iterable[Layer_Type] = (Layer_Type.EYE,),
) -> list[Action]:
    actions: list[Action] = []
    if grow:
        actions.extend(AddResLayer.generate_all_actions(model, layer_types=layer_types))
        actions.extend(AddResConvLayer.generate_all_actions(model))
        actions.extend(AddSeqLayer.generate_all_actions(model))
        actions.extend(AddSeqConvLayer.generate_all_actions(model))
    if shrink:
        actions.extend(DelLayer.generate_all_actions(model))
        actions.extend(DelNeurons.generate_all_actions(model))
    return actions
