"""Load job map and baselines from JSON next to the worker package."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS_PATH = ROOT / "jobs.json"
BASELINES_PATH = ROOT / "baselines.json"


def load_jobs(path: Path = JOBS_PATH) -> dict[str, dict[str, str]]:
    """Return job name -> {script, dataset}."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_baselines(path: Path = BASELINES_PATH) -> dict[str, dict[str, float]]:
    """Return dataset -> {mean_val_acc, max_params}."""
    return json.loads(path.read_text(encoding="utf-8"))


def script_path_for_job(job: str, jobs: dict[str, dict[str, str]] | None = None) -> str:
    """Return the checkout-relative script path for a registered job name."""
    registered = jobs if jobs is not None else load_jobs()
    if job not in registered:
        raise KeyError(f"unknown job {job!r}")
    return registered[job]["script"]
