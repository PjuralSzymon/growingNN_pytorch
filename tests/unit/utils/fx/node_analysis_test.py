"""Unit tests for ``growingnn.utils.fx.node_analysis``."""

import sys
from pathlib import Path

import pytest
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.fx import ModuleResolver
from tests.model_factory import ModelFactory


def test_find_call_module_raises_for_missing_target():
    """
    find_call_module should raise ValueError when no call_module matches the name.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())
    nodes = list(gm.graph.nodes)

    # Act / Assert
    assert ModuleResolver.find_call_module(nodes, "l1") is not None
    with pytest.raises(ValueError, match="No call_module node"):
        ModuleResolver.find_call_module(nodes, "res1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
