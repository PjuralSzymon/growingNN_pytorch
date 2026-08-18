"""Unit tests for JSON run storage used by the dashboard timeline."""

from pathlib import Path

from worker.storage import list_jobs, list_runs, load_job, read_log_tail, save_job, save_run


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


def test_save_job_and_list_jobs_newest_updated_first(tmp_path: Path):
    """
    save_job should persist status.json and list_jobs should return newest updated_at first.
    """

    # Arrange
    older = {
        "id": "jobold",
        "sha": "aaa",
        "pr": 1,
        "state": "done",
        "queued_at": "2026-08-01T10:00:00Z",
        "updated_at": "2026-08-01T10:00:00Z",
    }
    newer = {
        "id": "jobnew",
        "sha": "bbb",
        "pr": 2,
        "state": "queued",
        "queued_at": "2026-08-18T12:00:00Z",
        "updated_at": "2026-08-18T12:00:00Z",
    }

    # Act
    save_job(tmp_path, older)
    save_job(tmp_path, newer)
    listed = list_jobs(tmp_path)

    # Assert
    assert [row["id"] for row in listed] == ["jobnew", "jobold"]
    assert load_job(tmp_path, "jobnew")["state"] == "queued"


def test_read_log_tail_returns_last_lines(tmp_path: Path):
    """
    read_log_tail should return the last lines of the job log file.
    """

    # Arrange
    save_job(tmp_path, {"id": "joblog", "state": "running", "updated_at": "2026-08-18T12:00:00Z"})
    log_path = tmp_path / "jobs" / "joblog" / "log.txt"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    # Act
    tail = read_log_tail(tmp_path, "joblog", max_lines=2)

    # Assert
    assert tail == "two\nthree"
