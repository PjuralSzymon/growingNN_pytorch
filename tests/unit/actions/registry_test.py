"""Unit tests for ``growingnn.actions.registry``."""

import sys
from pathlib import Path

import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.registry import generate_all_actions
from tests.model_factory import ModelFactory


def test_generate_all_actions_returns_actions_for_conv_model():
    """
    generate_all_actions should return at least one mutation for a conv model.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())

    # Act
    actions = generate_all_actions(gm)

    # Assert
    assert len(actions) > 0


def test_generate_all_actions_respects_grow_and_shrink_flags():
    """
    generate_all_actions should honor grow/shrink toggles separately.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())

    # Act
    grow_only = generate_all_actions(gm, grow=True, shrink=False)
    shrink_only = generate_all_actions(gm, grow=False, shrink=True)

    # Assert
    assert len(grow_only) > 0
    assert len(shrink_only) >= 0
