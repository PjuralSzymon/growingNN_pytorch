import sys
from pathlib import Path

import torch.fx as fx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_layer import AddResLayer
from tests.model_factory import ModelFactory


"Generate AddResLayer actions for a simple linear chain"
def test_add_res_layer_generate_all_actions_linear_chain():
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    actions = AddResLayer.generate_all_actions(gm)

    # For l1->l2->l3, module_dependency_pairs yields 3 pairs; we create one action per Layer_Type.
    assert len(actions) == 3 * len(list(Layer_Type))

def test_add_res_layer_execute():
    #Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    executed_actions = []

    #Act
    for i in range(3):
        actions = AddResLayer.generate_all_actions(gm)
        for action in actions:
            action.execute(gm)
            executed_actions.append(action)

    #Assert
    assert gm.forward_blank() is not None
    for action in executed_actions:
        assert action.params[0] in gm.graph.nodes
        assert action.params[1] in gm.graph.nodes

if __name__ == "__main__":
    pytest.main([__file__, "-v"])