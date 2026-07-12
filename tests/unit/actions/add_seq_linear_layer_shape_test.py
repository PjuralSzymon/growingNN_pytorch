"""Shape-based action generation for ``AddSeqLinearLayer``."""

import torch
import torch.fx as fx

from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.core import config
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


def test_generate_all_actions_uses_shape_bridge_for_linear_chain():
    """
    AddSeqLinearLayer should propose one action per sequential pair when ShapeProp
    reports matching feature sizes along l1 -> l2 -> l3.
    """

    # Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    # Act
    actions = AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))

    # Assert
    assert len(actions) == 2
    for action in actions:
        layer = action.params[2]
        if isinstance(layer, torch.nn.Linear):
            assert layer.in_features > 0
            assert layer.out_features > 0


def test_generate_all_actions_skips_when_weight_matrix_exceeds_config_limit(monkeypatch):
    """
    AddSeqLinearLayer should skip pairs whose Linear weight matrix in*out exceeds config limit.
    """

    # Arrange
    monkeypatch.setattr(config, "MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE", 1)
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    actions = AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))

    # Assert
    assert actions == []


def test_generate_all_actions_proposes_plain_linear_between_conv_and_linear():
    """
    AddSeqLinearLayer should propose a bare Linear for conv->linear pairs (pool/flatten remain in the graph).
    """

    # Arrange
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)

    # Act
    actions = AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4, 32, 32)))

    # Assert
    conv_to_linear = [
        a
        for a in actions
        if isinstance(a.params[2], torch.nn.Linear)
        and a.params[0] == "c2"
        and a.params[1] == "l1"
    ]
    assert len(conv_to_linear) == 1
    assert conv_to_linear[0].params[2].in_features == conv_to_linear[0].params[2].out_features
