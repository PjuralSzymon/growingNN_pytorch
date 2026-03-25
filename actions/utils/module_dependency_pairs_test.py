import sys
from pathlib import Path

import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch.nn as nn

from growingnn.actions.utils.model_analyser import module_dependency_pairs


class ModelSimpleTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class ModelWithResidualSkip(nn.Module):
    """Main path l1→l2→l3 plus one skip branch l2→l4, merged with addition (residual-style).
    l1→l2→l3→l4
        |_____|
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 4)
        self.l4 = nn.Linear(4, 4)

    def forward(self, x):
        a = self.l1(x)
        b = self.l2(a)
        c = self.l3(b)
        d = self.l4(a)
        return c + d


@pytest.mark.description("Module dependency pairs should be correct for a linear chain of modules")
def test_module_dependency_pairs_linear_chain():
    # Arrange
    model = ModelSimpleTest()
    gm = fx.symbolic_trace(model)
    pairs = set(module_dependency_pairs(gm))

    # Act and Assert
    assert pairs == {
        ("l1", "l2"),
        ("l1", "l3"),
        ("l2", "l3"),
    }

@pytest.mark.description("With a residual branch, l1 also reaches l4 directly; pairs include (l1,l4) in addition to the chain.")
def test_module_dependency_pairs_with_residual_skip():
    # Arrange
    model = ModelWithResidualSkip()
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