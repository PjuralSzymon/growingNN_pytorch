from ast import Mod
import sys
from pathlib import Path

import torch.fx as fx


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from growingnn.utils.fx import ModuleClassifier, GraphStructureQuery
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph

from tests.model_factory import ModelFactory


"Module dependency pairs should be correct for a linear chain of modules"
def test_module_dependency_pairs_linear_chain():
    # Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    # Act and Assert
    assert pairs == {
        ("l1", "l2"),
    }


"Module dependency pairs should work around activation and batch normalization layers"
def test_module_dependency_pairs_linear_chain_with_activation():
    # Arrange
    model = ModelFactory.simple_chain_3_with_activation()
    gm = fx.symbolic_trace(model)

    # Act 
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    #Assert
    assert pairs == {
        ("l1", "l2"),
    }


"Module dependency pairs should work for deeply nested submodules"
def test_module_dependency_pairs_deeply_nested_submodules():
    # Arrange
    model = ModelFactory.deeply_nested_submodules()
    gm = fx.symbolic_trace(model)

    # Act 
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    #Assert
    assert len(pairs) == 10

"Module dependency pairs should avoid dependency pairs with activation and batch normalization layers"
def test_avoid_dependency_pairs_with_activation():
    # Arrange
    model_normal = ModelFactory.complex_residual_many_widths()
    model_activations = ModelFactory.complex_residual_many_widths_with_activation()
    gm_normal = fx.symbolic_trace(model_normal)
    gm_activations = fx.symbolic_trace(model_activations)

    # Act
    pairs_normal = set(GraphStructureQuery.module_dependency_pairs(gm_normal))
    pairs_activations = set(GraphStructureQuery.module_dependency_pairs(gm_activations))

    #Assert
    print("pairs_normal: %s", pairs_normal)
    print("pairs_activations: %s", pairs_activations)
    assert len(pairs_normal) == len(pairs_activations)

"With a residual branch, l1 also reaches l4 directly; pairs include (l1,l4) in addition to the chain."
def test_module_dependency_pairs_with_residual_skip():
    # Arrange
    model = ModelFactory.residual_skip()
    gm = fx.symbolic_trace(model)
    pairs = set(GraphStructureQuery.module_dependency_pairs(gm))

    assert pairs == {
        ("l1", "l2"),
        ("l1", "l3"),
        ("l1", "l4"),
        ("l2", "l3"),
    }


"Sequential pairs are only immediate module-to-module steps along the graph."
def test_module_sequential_pairs_linear_chain():
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    assert set(GraphStructureQuery.module_sequential_pairs(gm)) == {("l1", "l2"), ("l2", "l3")}


def test_module_sequential_pairs_with_residual_skip():
    model = ModelFactory.residual_skip()
    gm = fx.symbolic_trace(model)
    assert set(GraphStructureQuery.module_sequential_pairs(gm)) == {
        ("l1", "l2"),
        ("l1", "l4"),
        ("l2", "l3"),
    }

"is_hidden_module should return True only for modules with both upstream and downstream modules."
def test_is_hidden_module_true_for_middle_module():
    # Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    l2_node = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l2")

    # Act
    result = ModuleClassifier.is_hidden_module(l2_node)

    # Assert
    assert result is True


def test_is_edge_into_hidden_module_accepts_visible_or_hidden_to_hidden():
    """
    ModuleClassifier.is_edge_into_hidden_module is true for visible→hidden and hidden→hidden only.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())
    l1 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l1")
    l2 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l2")
    l3 = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l3")

    # Act & Assert
    assert ModuleClassifier.is_edge_into_hidden_module(l1, l2) is True
    assert ModuleClassifier.is_edge_into_hidden_module(l2, l3) is False
    assert ModuleClassifier.is_edge_into_hidden_module(l2, l1) is False
    assert ModuleClassifier.is_edge_into_hidden_module(l1, l3) is False


"get_all_modules should return only hidden modules for a linear chain."
def test_get_all_modules_returns_only_hidden_linear_chain_modules():
    # Arrange
    model = ModelFactory.simple_chain_3()

    # Act
    result = GraphStructureQuery.get_all_hidden_modules(model)

    # Assert
    assert result == ["l2"]


"get_all_modules should return only hidden modules in a mixed conv/linear pipeline."
def test_get_all_modules_returns_only_hidden_conv_chain_modules():
    # Arrange
    model = ModelFactory.simple_conv_chain_2()

    # Act
    result = GraphStructureQuery.get_all_hidden_modules(model)

    # Assert
    assert result == ["c2", "pool", "l1"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])