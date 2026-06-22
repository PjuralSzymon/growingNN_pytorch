"""Unit tests for safe experiment path resolution."""

import sys
from pathlib import Path

import pytest

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


def test_resolve_experiment_directory_rejects_path_outside_root(tmp_path):
    """
    resolve_experiment_directory should reject traversal outside the experiments root.
    """
    # Arrange
    root = tmp_path / "experiments"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    # Act / Assert
    with pytest.raises(ValueError, match="outside allowed root"):
        resolve_experiment_directory(str(outside), root=root)


def test_resolve_experiment_directory_rejects_parent_traversal(tmp_path):
    """
    resolve_experiment_directory should reject .. segments that escape the root.
    """
    # Arrange
    root = tmp_path / "experiments"
    root.mkdir()

    # Act / Assert
    with pytest.raises(ValueError, match="outside allowed root"):
        resolve_experiment_directory("../outside", root=root)
