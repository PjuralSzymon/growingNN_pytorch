"""Shape-based action generation for ``AddSeqLayer``."""

import torch
import torch.fx as fx

from growingnn.actions.add_seq_layer import AddSeqLayer
from tests.model_factory import ModelFactory


def test_generate_all_actions_uses_shape_bridge_for_linear_chain():
    """
    AddSeqLayer should propose one action per sequential pair when ShapeProp
    reports matching feature sizes along l1 -> l2 -> l3.
    """

    # Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    # Act
    actions = AddSeqLayer.generate_all_actions(gm)

    # Assert
    assert len(actions) == 2
    for action in actions:
        layer = action.params[2]
        assert layer.in_features > 0
        assert layer.out_features > 0
