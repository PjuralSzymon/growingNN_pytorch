"""Safe JSON reads for partially written experiment files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from growingnn_board.schemas import MainExperiment, TrainingMetrics


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def read_main(path: Path) -> MainExperiment | None:
    raw = read_json(path)
    if raw is None:
        return None
    try:
        return MainExperiment.model_validate(raw)
    except ValidationError:
        return None


def read_training_metrics(path: Path) -> TrainingMetrics | None:
    raw = read_json(path)
    if raw is None:
        return None
    try:
        return TrainingMetrics.model_validate(raw)
    except ValidationError:
        return None


def resolve_experiment_directory(path: str, *, root: Path) -> Path:
    """Resolve an experiment directory under root; reject paths outside root."""
    experiments_root = root.resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (experiments_root / candidate).resolve()
    try:
        resolved.relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError("Experiment path outside allowed root") from exc
    return resolved


def directory_status(last_update_iso: str) -> str:
    from datetime import datetime, timezone

    try:
        ts = datetime.fromisoformat(last_update_iso.replace("Z", "+00:00"))
    except ValueError:
        return "inactive"
    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    if age_hours < 1:
        return "active"
    if age_hours < 6:
        return "recent"
    return "inactive"
