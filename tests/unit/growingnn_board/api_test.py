"""Unit tests for GrowingNN Board simulation preview API routes."""

from __future__ import annotations

from unittest.mock import patch

import growingnn_board.api as board_api
from growingnn_board.cache import ExperimentCache


def _simulation() -> dict:
    candidate = {
        "action": "action-a",
        "name": "Action A",
        "chosen": True,
        "scoreBreakdown": {
            "terms": {
                "accuracy": {"raw": 0.8},
                "paramCount": {"raw": 0.2},
            }
        },
    }
    return {
        "actionChosen": "action-a",
        "candidates": [candidate],
        "searchTree": {
            "id": "0",
            "depth": 0,
            "children": [
                {
                    "id": "0-0",
                    "depth": 1,
                    "action": "action-a",
                    "finalScore": 0.8,
                    "children": [],
                }
            ],
        },
    }


def test_recalculate_simulation_scores_returns_lightweight_preview(monkeypatch):
    """
    recalculate_simulation_scores should return recalculated candidates without changing the cache.
    """
    # Arrange
    simulation = _simulation()
    monkeypatch.setattr(
        board_api,
        "_cache",
        ExperimentCache(simulations={2: simulation}),
    )

    # Act
    result = board_api.recalculate_simulation_scores(2, accuracy_weight=1.0, param_count_weight=1.0)

    # Assert
    assert result["candidates"][0]["score"] == 0.5
    assert board_api._cache.simulations[2]["searchTree"]["children"][0]["finalScore"] == 0.8


def test_get_simulation_search_tree_renders_recalculated_top_level_score(tmp_path, monkeypatch):
    """
    get_simulation_search_tree should render a preview tree when both alternative weights are supplied.
    """
    # Arrange
    monkeypatch.setattr(
        board_api,
        "_cache",
        ExperimentCache(path=tmp_path, simulations={2: _simulation()}),
    )

    # Act
    with patch("growingnn_board.api.render_search_tree_html", return_value="<html></html>") as render:
        board_api.get_simulation_search_tree(2, accuracy_weight=1.0, param_count_weight=1.0)

    # Assert
    rendered_tree = render.call_args.args[0]
    assert rendered_tree["children"][0]["finalScore"] == 0.5
