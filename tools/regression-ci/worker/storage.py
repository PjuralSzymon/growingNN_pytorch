"""Save and list regression CI run records as JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_run(runs_dir: Path, record: dict[str, Any]) -> Path:
    """Write one finished (or failed) job record to the runs volume."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(record["finished_at"]).replace(":", "-")
    pr = record.get("pr")
    pr_part = f"pr-{pr}" if pr is not None else "pr-none"
    path = runs_dir / f"{stamp}_{pr_part}_{record['job']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Return run records newest first. Missing folders yield an empty list."""
    if not runs_dir.is_dir():
        return []
    files = sorted(runs_dir.glob("*.json"), reverse=True)
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]
