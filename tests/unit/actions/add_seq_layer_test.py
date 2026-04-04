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
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.model_factory import ModelFactory
from tests.regression.regression_utils import FOLDER_NAME, clear_regression_folder, parse_regression_cli
from growingnn.actions.add_seq_layer import AddSeqLayer



def test_add_res_layer_generate_all_actions_linear_chain():
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    actions = AddSeqLayer.generate_all_actions(gm)

    # For l1->l2->l3, module_dependency_pairs yields 3 pairs; we create one action per Layer_Type.
    assert len(actions) == len(list(Layer_Type)) - 1

def test_add_seq_layer_execute():
    #Arrange
    args = parse_regression_cli()
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    initial_amount_of_linears = sum(1 for m in gm.modules() if isinstance(m, torch.nn.Linear))

    # Act
    for _ in range(30):
        actions: List[AddSeqLayer] = AddSeqLayer.generate_all_actions(gm)
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