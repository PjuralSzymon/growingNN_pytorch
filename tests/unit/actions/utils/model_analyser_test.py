import sys
from pathlib import Path

import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from growingnn.actions.utils.model_analyser import (
    _is_hidden_module,
    get_all_hidden_modules,
    module_dependency_pairs,
    module_sequential_pairs,
)
from tests.model_factory import ModelFactory


"Module dependency pairs should be correct for a linear chain of modules"
def test_module_dependency_pairs_linear_chain():
    # Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    pairs = set(module_dependency_pairs(gm))

    # Act and Assert
    assert pairs == {
        ("l1", "l2"),
        ("l2", "l3"),
    }

"With a residual branch, l1 also reaches l4 directly; pairs include (l1,l4) in addition to the chain."
def test_module_dependency_pairs_with_residual_skip():
    # Arrange
    model = ModelFactory.residual_skip()
    gm = fx.symbolic_trace(model)
    pairs = set(module_dependency_pairs(gm))

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
    assert set(module_sequential_pairs(gm)) == {("l1", "l2"), ("l2", "l3")}


def test_module_sequential_pairs_with_residual_skip():
    model = ModelFactory.residual_skip()
    gm = fx.symbolic_trace(model)
    assert set(module_sequential_pairs(gm)) == {
        ("l1", "l2"),
        ("l1", "l4"),
        ("l2", "l3"),
    }

"_is_hidden_module should return True only for modules with both upstream and downstream modules."
def test_is_hidden_module_true_for_middle_module():
    # Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    l2_node = next(n for n in gm.graph.nodes if n.op == "call_module" and n.target == "l2")

    # Act
    result = _is_hidden_module(l2_node)

    # Assert
    assert result is True


"get_all_modules should return only hidden modules for a linear chain."
def test_get_all_modules_returns_only_hidden_linear_chain_modules():
    # Arrange
    model = ModelFactory.simple_chain_3()

    # Act
    result = get_all_hidden_modules(model)

    # Assert
    assert result == ["l2"]


"get_all_modules should return only hidden modules in a mixed conv/linear pipeline."
def test_get_all_modules_returns_only_hidden_conv_chain_modules():
    # Arrange
    model = ModelFactory.simple_conv_chain_2()

    # Act
    result = get_all_hidden_modules(model)

    # Assert
    assert result == ["c2", "pool", "l1"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])