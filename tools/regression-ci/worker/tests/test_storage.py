"""Unit tests for JSON run storage used by the dashboard timeline."""

from pathlib import Path

from worker.storage import list_runs, save_run


def test_save_run_and_list_runs_newest_first(tmp_path: Path):
    """
    save_run should write one JSON file and list_runs should return newest first.
    """

    # Arrange
    older = {
        "job": "mnist",
        "finished_at": "2026-08-01T10-00-00Z",
        "pr": 1,
        "ok": True,
    }
    newer = {
        "job": "mnist",
        "finished_at": "2026-08-18T12-00-00Z",
        "pr": 2,
        "ok": True,
    }

    # Act
    save_run(tmp_path, older)
    save_run(tmp_path, newer)
    listed = list_runs(tmp_path)

    # Assert
    assert [row["pr"] for row in listed] == [2, 1]


def test_list_runs_returns_empty_when_folder_missing(tmp_path: Path):
    """
    list_runs should return [] when the runs volume has not been created yet.
    """

    # Arrange
    missing = tmp_path / "runs"

    # Act
    listed = list_runs(missing)

    # Assert
    assert listed == []
