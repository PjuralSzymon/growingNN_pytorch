"""Unit tests for Experiment 006 chart helpers."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from documentation.website.scripts.generate_experiment_006_charts import (
    plot_mean_simulation_scores,
    summarize_action_scores,
)


def test_summarize_action_scores_reports_mean_n_min_max():
    """
    summarize_action_scores should return n, mean, min, and max for each label.
    """

    # Arrange
    scores = {
        "Add Neurons Action": [0.2, 0.4, 0.6],
        "Add Res Conv Layer Action": [1.0],
    }

    # Act
    summary = summarize_action_scores(scores)

    # Assert
    neuron = summary["Add Neurons Action"]
    assert neuron["n"] == 3
    assert neuron["mean"] == pytest.approx(0.4)
    assert neuron["min"] == 0.2
    assert neuron["max"] == 0.6
    assert summary["Add Res Conv Layer Action"]["n"] == 1
    assert summary["Add Res Conv Layer Action"]["mean"] == pytest.approx(1.0)
    assert "Delete Neurons Action" not in summary


def test_plot_mean_simulation_scores_writes_png(tmp_path: Path):
    """
    plot_mean_simulation_scores should write the mean-score chart from group stats.
    """

    # Arrange
    analysis = {
        "groups": {
            "none": {
                "mean_score_by_action": {
                    "Add Res Conv Layer Action": {"n": 2, "mean": 0.7, "min": 0.6, "max": 0.8},
                }
            },
            "add11_del01": {
                "mean_score_by_action": {
                    "Add Neurons Action": {"n": 2, "mean": 0.5, "min": 0.4, "max": 0.6},
                    "Add Res Conv Layer Action": {"n": 2, "mean": 0.65, "min": 0.6, "max": 0.7},
                }
            },
            "add15_del05": {"mean_score_by_action": {}},
            "add20_del09": {"mean_score_by_action": {}},
        }
    }

    # Act
    plot_mean_simulation_scores(analysis, tmp_path)

    # Assert
    assert (tmp_path / "006-mean-simulation-score-by-action.png").is_file()
