import sys
from pathlib import Path

import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from growingnn.actions.utils.model_analyser import module_dependency_pairs
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
        ("l1", "l3"),
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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])