"""Unit tests for safe experiment path resolution."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn_board.file_reader import resolve_experiment_directory


def test_resolve_experiment_directory_accepts_relative_path_under_root(tmp_path):
    """
    resolve_experiment_directory should map a relative path under the experiments root.
    """
    # Arrange
    root = tmp_path / "experiments"
    experiment = root / "run_a" / "board"
    experiment.mkdir(parents=True)

    # Act
    resolved = resolve_experiment_directory("run_a/board", root=root)

    # Assert
    assert resolved == experiment.resolve()


def test_resolve_experiment_directory_accepts_absolute_path_outside_root(tmp_path):
    """
    resolve_experiment_directory should accept an absolute directory outside the recent-experiments root.
    """
    # Arrange
    root = tmp_path / "experiments"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    # Act
    resolved = resolve_experiment_directory(str(outside), root=root)

    # Assert
    assert resolved == outside.resolve()


def test_resolve_experiment_directory_searches_for_valid_main_json(tmp_path):
    """
    resolve_experiment_directory should find a nested experiment definition below the supplied directory.
    """
    # Arrange
    root = tmp_path / "experiments"
    root.mkdir()
    experiment = tmp_path / "selected" / "run_a" / "board"
    experiment.mkdir(parents=True)
    (experiment / "main.json").write_text(
        json.dumps(
            {
                "experimentId": "run-a-id",
                "experimentName": "run_a",
                "experimentStartedOn": "2026-07-15T19:00:00Z",
                "status": "running",
                "lastUpdate": "2026-07-15T20:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    # Act
    resolved = resolve_experiment_directory(str(tmp_path / "selected"), root=root)

    # Assert
    assert resolved == experiment.resolve()
