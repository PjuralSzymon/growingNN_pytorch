"""Static integration tests for the Training metrics chart grid."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINING_PAGE = _REPO_ROOT / "tools" / "growingnn_board" / "static" / "pages" / "training"


def test_training_page_contains_six_charts_in_comparison_order():
    """
    The Training page should arrange six chart canvases in the planned 2x3 comparison order.
    """
    # Arrange
    html = (_TRAINING_PAGE / "training.html").read_text(encoding="utf-8")
    expected_ids = [
        "chart-train-acc",
        "chart-val-acc",
        "chart-param-count",
        "chart-train-loss",
        "chart-val-loss",
        "chart-learning-rate",
    ]

    # Act
    canvas_ids = re.findall(r'<canvas id="(chart-[^"]+)"', html)

    # Assert
    assert canvas_ids == expected_ids


def test_training_chart_renderer_uses_existing_parameter_and_learning_rate_fields():
    """
    The Training chart renderer should source both new series from existing epoch metric fields.
    """
    # Arrange
    javascript = (_TRAINING_PAGE / "training.js").read_text(encoding="utf-8")

    # Act
    parameter_series = '"chart-param-count",\n    "paramCount"' in javascript
    learning_rate_series = '"chart-learning-rate",\n    "learningRate"' in javascript

    # Assert
    assert parameter_series is True
    assert learning_rate_series is True
