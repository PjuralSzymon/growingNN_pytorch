import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.fx as fx
import pytest

from growingnn.actions.utils.model_transformations import add_new_residual_layer, _find_call_module


class ModelSimpleTest(nn.Module): 
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

@pytest.mark.description("Finding a non-existing layer should raise a ValueError")
def test_finding_non_existing_layer_should_raise_value_error():
    # Arrange
    model = ModelSimpleTest() 
    not_existing_layer_name = "res1"
    existing_layer_name = "l1"
    gm = fx.symbolic_trace(model)
    nodes = list(gm.graph.nodes)

    # Act and Assert
    assert _find_call_module(nodes, existing_layer_name) is not None
    with pytest.raises(ValueError, match="No call_module node"):
        _find_call_module(nodes, not_existing_layer_name)

@pytest.mark.description("Residual branch uses zero weights, so output should match the graph before the edit.")
def test_adding_residual_layer_without_change():
    # Arrange
    model = ModelSimpleTest() 
    x = torch.randn(1, 4)
    gm = fx.symbolic_trace(model) 
    y_initial = gm(x)
    layer = nn.Linear(4, 4) 
    layer.weight.data.zero_() 
    layer.bias.data.zero_() 

    #Act
    add_new_residual_layer(gm, "l1", "l2", layer, name="res1")
    y_after = gm(x)

    #Assert
    assert torch.allclose(y_after, y_initial)

@pytest.mark.description("Adding a residual layer should add a new module to the graph")
def test_adding_residual_layer_should_add_new_module_to_graph():
    # Arrange
    model = ModelSimpleTest() 
    new_layer_name = "res1"
    gm = fx.symbolic_trace(model) 

    #Act
    add_new_residual_layer(gm, "l1", "l2", nn.Linear(4, 4), name=new_layer_name)
    nodes = list(gm.graph.nodes)

    #Assert
    assert _find_call_module(nodes, new_layer_name) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])