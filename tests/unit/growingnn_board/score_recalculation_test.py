"""Unit tests for board-only simulation score previews."""

from __future__ import annotations

import copy

import pytest

from growingnn_board.score_recalculation import apply_recalculated_scores, recalculate_simulation


def _simulation() -> dict:
    return {
        "actionChosen": "action-a",
        "candidates": [
            {
                "action": "action-a",
                "name": "Action A",
                "chosen": True,
                "scoreBreakdown": {
                    "terms": {
                        "accuracy": {"weight": 1.0, "raw": 0.9, "weighted": 0.9},
                        "paramCount": {"weight": 0.1, "raw": 0.1, "weighted": 0.01},
                    }
                },
            },
            {
                "action": "action-b",
                "name": "Action B",
                "chosen": False,
                "scoreBreakdown": {
                    "terms": {
                        "accuracy": {"weight": 1.0, "raw": 0.8, "weighted": 0.8},
                        "paramCount": {"weight": 0.1, "raw": 1.0, "weighted": 0.1},
                    }
                },
            },
        ],
    }


def test_recalculate_simulation_uses_new_weights_with_saved_raw_terms():
    """
    recalculate_simulation should apply new multipliers to unchanged saved raw terms.
    """
    # Arrange
    simulation = _simulation()

    # Act
    result = recalculate_simulation(simulation, accuracy_weight=1.0, param_count_weight=1.0)

    # Assert
    assert result["candidates"][0]["score"] == pytest.approx(0.5)


def test_recalculate_simulation_does_not_mutate_saved_candidate_data():
    """
    recalculate_simulation should leave the original simulation data unchanged.
    """
    # Arrange
    simulation = _simulation()
    original = copy.deepcopy(simulation)

    # Act
    recalculate_simulation(simulation, accuracy_weight=0.2, param_count_weight=0.8)

    # Assert
    assert simulation == original


def test_recalculate_simulation_reports_changed_projected_action():
    """
    recalculate_simulation should report when the highest new composite differs from the saved choice.
    """
    # Arrange
    simulation = _simulation()

    # Act
    result = recalculate_simulation(simulation, accuracy_weight=0.1, param_count_weight=1.0)

    # Assert
    assert result["projectedAction"] == "action-b"
    assert result["sameAction"] is False


@pytest.mark.parametrize(
    ("accuracy_weight", "param_count_weight"),
    [(-1.0, 1.0), (float("inf"), 1.0), (0.0, 0.0)],
)
def test_recalculate_simulation_rejects_invalid_weights(accuracy_weight, param_count_weight):
    """
    recalculate_simulation should reject negative, non-finite, and all-zero weights.
    """
    # Arrange
    simulation = _simulation()

    # Act / Assert
    with pytest.raises(ValueError):
        recalculate_simulation(simulation, accuracy_weight, param_count_weight)


def test_recalculate_simulation_reports_action_with_missing_active_term():
    """
    recalculate_simulation should report an action that lacks a raw term with a positive weight.
    """
    # Arrange
    simulation = _simulation()
    del simulation["candidates"][0]["scoreBreakdown"]["terms"]["paramCount"]

    # Act
    result = recalculate_simulation(simulation, accuracy_weight=1.0, param_count_weight=1.0)

    # Assert
    assert result["unavailableActions"] == ["Action A"]


def test_apply_recalculated_scores_updates_only_top_level_action_node():
    """
    apply_recalculated_scores should replace a matching root child score but retain its descendant score.
    """
    # Arrange
    tree = {
        "id": "0",
        "children": [
            {
                "id": "0-0",
                "action": "action-a",
                "finalScore": 0.1,
                "compositeScore": 0.1,
                "children": [
                    {
                        "id": "0-0-0",
                        "action": "action-b",
                        "finalScore": 0.3,
                        "compositeScore": 0.3,
                        "children": [],
                    }
                ],
            }
        ],
    }
    recalculation = {"candidates": [{"action": "action-a", "score": 0.8}]}

    # Act
    result = apply_recalculated_scores(tree, recalculation)

    # Assert
    assert result["children"][0]["finalScore"] == 0.8
    assert result["children"][0]["children"][0]["finalScore"] == 0.3


def test_apply_recalculated_scores_does_not_mutate_saved_tree():
    """
    apply_recalculated_scores should return a copy and keep the saved tree unchanged.
    """
    # Arrange
    tree = {"id": "0", "children": [{"id": "0-0", "action": "action-a", "finalScore": 0.1}]}

    # Act
    apply_recalculated_scores(tree, {"candidates": [{"action": "action-a", "score": 0.8}]})

    # Assert
    assert tree["children"][0]["finalScore"] == 0.1
