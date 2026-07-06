import random
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.core import config
from tests.model_factory import ModelFactory


def test_generate_all_actions_skips_when_weight_matrix_exceeds_config_limit(monkeypatch):
    """
    AddResLinearLayer should skip pairs whose Linear weight matrix in*out exceeds config limit.
    """

    # Arrange
    monkeypatch.setattr(config, "MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE", 1)
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    actions = AddResLinearLayer.generate_all_actions(gm)

    # Assert
    assert actions == []


"Generate AddResLinearLayer actions for a simple linear chain"
def test_add_res_linear_layer_generate_all_actions_linear_chain():
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    actions = AddResLinearLayer.generate_all_actions(gm)

    # For l1->l2->l3, only l1->l2 is an edge into a hidden module; one action per Layer_Type.
    assert len(actions) == len(list(Layer_Type))

def test_add_res_linear_layer_execute():
    #Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    initial_amount_of_linears = sum(1 for m in gm.modules() if isinstance(m, torch.nn.Linear))

    # Act
    for _ in range(30):
        actions: List[AddResLinearLayer] = AddResLinearLayer.generate_all_actions(gm)
        idx = rng.randrange(len(actions))
        actions[idx].execute(gm)
    out = gm(x)

    # Assert
    num_linears = sum(1 for m in gm.modules() if isinstance(m, torch.nn.Linear))
    assert num_linears == 30 + initial_amount_of_linears
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
    for action in executed_actions:
        assert action.params[0] in gm.graph.nodes
        assert action.params[1] in gm.graph.nodes

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
