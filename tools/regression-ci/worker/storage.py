"""Save finished run JSON and live job status for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOG_TAIL_LINES = 200


def save_run(runs_dir: Path, record: dict[str, Any]) -> Path:
    """Write one finished (or failed) dataset record to the runs volume."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(record["finished_at"]).replace(":", "-")
    pr = record.get("pr")
    pr_part = f"pr-{pr}" if pr is not None else "pr-none"
    path = runs_dir / f"{stamp}_{pr_part}_{record['job']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Return finished run records newest first. Missing folders yield []."""
    if not runs_dir.is_dir():
        return []
    files = sorted(runs_dir.glob("*.json"), reverse=True)
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def jobs_root(runs_dir: Path) -> Path:
    return runs_dir / "jobs"


def job_dir(runs_dir: Path, job_id: str) -> Path:
    return jobs_root(runs_dir) / job_id


def job_log_path(runs_dir: Path, job_id: str) -> Path:
    return job_dir(runs_dir, job_id) / "log.txt"


def save_job(runs_dir: Path, record: dict[str, Any]) -> Path:
    """Write the live job status so a dashboard refresh can see queued work."""
    path = job_dir(runs_dir, str(record["id"])) / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_job(runs_dir: Path, job_id: str) -> dict[str, Any] | None:
    """Return one job status file, or None when it has not been written."""
    path = job_dir(runs_dir, job_id) / "status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs(runs_dir: Path) -> list[dict[str, Any]]:
    """Return job statuses newest-updated first, including queued and running."""
    root = jobs_root(runs_dir)
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in root.glob("*/status.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(
        key=lambda row: str(row.get("updated_at") or row.get("queued_at") or ""),
        reverse=True,
    )
    return records


def read_log_tail(
    runs_dir: Path, job_id: str, max_lines: int = LOG_TAIL_LINES
) -> str:
    """Return the last lines of the job log, or empty when none exists."""
    path = job_log_path(runs_dir, job_id)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
