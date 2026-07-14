"""Unit tests for ``growingnn.actions.add_seq_dropout_layer``."""

import sys
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer
from growingnn.actions.utils.seq_insertion import iter_seq_shape_matched_pairs
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


def test_iter_seq_shape_matched_pairs_finds_linear_chain_edges():
    """
    iter_seq_shape_matched_pairs should list sequential edges with equal probed shapes.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))

    # Act
    pairs = {(cand.from_id, cand.to_id) for cand in iter_seq_shape_matched_pairs(TracedModel.create(gm, (1, 4)))}

    # Assert
    assert ("l1", "l2") in pairs
    assert ("l2", "l3") in pairs


def test_generate_all_actions_proposes_dropout_on_linear_chain():
    """
    AddSeqDropoutLayer.generate_all_actions should propose dropout inserts on matching pairs.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))

    # Act
    actions = AddSeqDropoutLayer.generate_all_actions(TracedModel.create(gm, (1, 4)), p=0.2)

    # Assert
    assert len(actions) >= 2
    assert all(isinstance(action.params[2], nn.Dropout) for action in actions)
    assert all(action.params[2].p == 0.2 for action in actions)


def test_execute_inserts_dropout_without_changing_eval_output_shape():
    """
    AddSeqDropoutLayer.execute should insert dropout and preserve eval-mode output shape.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))
    gm.eval()
    x = torch.randn(2, 4)
    y0 = gm(x)
    actions = AddSeqDropoutLayer.generate_all_actions(TracedModel.create(gm, (1, 4)), p=0.5)
    action = next(a for a in actions if a.params[0] == "l1" and a.params[1] == "l2")

    # Act
    action.execute(TracedModel.create(gm, (1, 4)))
    y1 = gm(x)

    # Assert
    assert any(isinstance(m, nn.Dropout) for m in gm.modules())
    assert y0.shape == y1.shape == (2, 10)


def test_generate_all_actions_uses_dropout2d_on_conv_chain():
    """
    AddSeqDropoutLayer should propose Dropout2d between conv layers with matching 4D shapes.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_conv_chain_2())

    # Act
    actions = AddSeqDropoutLayer.generate_all_actions(TracedModel.create(gm, (1, 4, 32, 32)), p=0.1)
    conv_actions = [a for a in actions if a.params[0] == "c1" and a.params[1] == "c2"]

    # Assert
    assert len(conv_actions) == 1
    assert isinstance(conv_actions[0].params[2], nn.Dropout2d)


def test_generate_all_actions_skips_pairs_adjacent_to_existing_dropout():
    """
    AddSeqDropoutLayer should not insert directly before or after another dropout layer.
    """
    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3(neurons=10))
    first = next(
        a for a in AddSeqDropoutLayer.generate_all_actions(TracedModel.create(gm, (1, 4)), p=0.2)
        if a.params[0] == "l1" and a.params[1] == "l2"
    )
    first.execute(TracedModel.create(gm, (1, 4)))
    dropout_ids = {
        name for name, mod in gm.named_modules()
        if isinstance(mod, (nn.Dropout, nn.Dropout2d))
    }

    # Act
    actions = AddSeqDropoutLayer.generate_all_actions(TracedModel.create(gm, (1, 4)), p=0.2)

    # Assert
    assert len(dropout_ids) == 1
    for action in actions:
        assert action.params[0] not in dropout_ids
        assert action.params[1] not in dropout_ids
