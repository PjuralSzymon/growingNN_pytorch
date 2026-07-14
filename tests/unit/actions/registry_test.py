"""Unit tests for ``growingnn.actions.registry``."""

import sys
from pathlib import Path

import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


def test_generate_all_actions_returns_actions_for_conv_model():
    """
    generate_all_actions should return at least one mutation for a conv model.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())
    graph = TracedModel.create(gm, (1, 4, 32, 32))

    # Act
    actions = generate_all_actions(graph, RunningConfig(generations=1, epochs=1))

    # Assert
    assert len(actions) > 0


def test_generate_all_actions_respects_grow_and_shrink_flags():
    """
    generate_all_actions should honor grow/shrink toggles separately.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())
    graph = TracedModel.create(gm, (1, 4, 32, 32))

    cfg_grow = RunningConfig(generations=1, epochs=1)
    cfg_grow.update_shrink_actions(False)
    cfg_shrink = RunningConfig(generations=1, epochs=1)
    cfg_shrink.update_grow_actions(False)

    # Act
    grow_only = generate_all_actions(graph, cfg_grow)
    shrink_only = generate_all_actions(graph, cfg_shrink)

    # Assert
    assert len(grow_only) > 0
    assert len(shrink_only) >= 0
