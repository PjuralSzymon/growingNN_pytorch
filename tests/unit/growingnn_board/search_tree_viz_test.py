"""Unit tests for board-side search tree HTML rendering."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from growingnn_board.search_tree_viz import (
    _node_color,
    _render_search_tree_html_fallback,
    render_search_tree_html,
    resolve_search_tree,
    tree_from_candidates,
)

def _sample_tree() -> dict:
    return {
        "id": "0",
        "name": "root",
        "depth": 0,
        "visits": 5,
        "children": [
            {
                "id": "0-0",
                "name": "Add Layer Action",
                "action": " ( Add Layer Action: [] ) ",
                "depth": 1,
                "visits": 2,
                "finalScore": 0.65,
                "meanScore": 0.65,
                "ucbScore": 1.1,
                "compositeScore": 0.65,
                "accuracyAfter": 0.58,
                "chosen": True,
                "maxDepthBelow": 2,
                "children": [
                    {
                        "id": "0-0-0",
                        "name": "Delete Neurons Action",
                        "action": " ( Delete Neurons Action: [] ) ",
                        "depth": 2,
                        "visits": 1,
                        "finalScore": 0.7,
                        "meanScore": 0.7,
                        "ucbScore": 0.9,
                        "compositeScore": 0.7,
                        "accuracyAfter": 0.61,
                        "chosen": False,
                        "maxDepthBelow": 2,
                        "children": [],
                    }
                ],
            }
        ],
    }


def test_tree_from_candidates_builds_flat_tree():
    """
    tree_from_candidates should build one-level searchTree JSON from candidate rows.
    """
    # Arrange
    candidates = [
        {
            "action": " ( Add Layer Action: [] ) ",
            "name": "Add Layer Action",
            "visits": 1,
            "score": 0.5,
            "compositeScore": 0.48,
            "ucbScore": 1.2,
            "chosen": True,
        }
    ]

    # Act
    tree = tree_from_candidates(candidates, rollouts=5)

    # Assert
    assert tree["visits"] == 5
    assert len(tree["children"]) == 1
    assert tree["children"][0]["finalScore"] == 0.48


def test_resolve_search_tree_falls_back_to_candidates():
    """
    resolve_search_tree should use candidates when searchTree is missing from simulation JSON.
    """
    # Arrange
    sim = {"rollouts": 2, "candidates": [{"action": "x", "name": "X", "score": 0.1, "visits": 1}]}

    # Act
    tree = resolve_search_tree(sim)

    # Assert
    assert tree is not None
    assert len(tree["children"]) == 1


@pytest.mark.parametrize(
    ("final_score", "expected_color"),
    [
        (0.0, "hsl(0, 70%, 88%)"),
        (0.5, "hsl(60, 70%, 88%)"),
        (1.0, "hsl(120, 70%, 88%)"),
    ],
)
def test_node_color_maps_final_score_to_red_yellow_green(final_score, expected_color):
    """
    _node_color should map the 0–1 final score range from red through yellow to green.
    """
    # Arrange
    node = {"id": "0-0", "depth": 1, "finalScore": final_score}

    # Act
    color = _node_color(node)

    # Assert
    assert color == expected_color


def test_node_color_keeps_nodes_without_final_score_neutral():
    """
    _node_color should keep a non-root node neutral when its final score is unavailable.
    """
    # Arrange
    node = {"id": "0-0", "depth": 1, "finalScore": None}

    # Act
    color = _node_color(node)

    # Assert
    assert color == "#ffffff"


def test_render_search_tree_html_fallback_without_pyvis():
    """
    render_search_tree_html should use plain HTML when pyvis is unavailable.
    """
    # Arrange
    tree = _sample_tree()

    # Act
    with patch(
        "growingnn_board.search_tree_viz._render_search_tree_html_pyvis",
        side_effect=ImportError("pyvis"),
    ):
        html = render_search_tree_html(tree, rollouts=5, max_depth=2)

    # Assert
    assert "gnn-tree" in html
    assert "Delete Neurons Action" in html
    assert "pip install pyvis" in html


def test_render_search_tree_html_fallback_direct():
    """
    _render_search_tree_html_fallback should render depth rows so nodes at the same depth share a row.
    """
    # Arrange
    tree = _sample_tree()

    # Act
    html = _render_search_tree_html_fallback(tree, rollouts=5, max_depth=2)

    # Assert
    assert "gnnNodeDetails" in html
    assert 'data-depth="1"' in html
    assert 'data-depth="2"' in html
    assert "Delete Neurons Action" in html
    assert 'style="background:hsl(78, 70%, 88%)"' in html


def test_render_search_tree_html_returns_vis_network_when_pyvis_available():
    """
    render_search_tree_html should return vis.js HTML when pyvis is installed.
    """
    pytest.importorskip("pyvis")
    # Arrange
    tree = _sample_tree()

    # Act
    html = render_search_tree_html(tree, rollouts=5, max_depth=2)

    # Assert
    if "vis.Network" in html:
        assert 'network.on("click"' in html
    else:
        assert "gnn-tree" in html
    assert "Delete Neurons Action" in html
