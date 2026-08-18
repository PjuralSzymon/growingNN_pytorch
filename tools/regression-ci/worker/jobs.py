"""Discover regression CI scripts in the checked-out repo."""

from __future__ import annotations

import json
from pathlib import Path

CI_SCRIPTS_DIR = Path("tests") / "regression" / "ci"
ROOT = Path(__file__).resolve().parents[1]
BASELINES_PATH = ROOT / "baselines.json"


def load_baselines(path: Path = BASELINES_PATH) -> dict[str, dict[str, float]]:
    """Return dataset -> {mean_val_acc, max_params}."""
    return json.loads(path.read_text(encoding="utf-8"))


def discover_ci_scripts(repo_dir: Path) -> list[tuple[str, str]]:
    """
    Return (job_name, checkout-relative path) for each script in tests/regression/ci.

    Skips underscore files such as __init__.py. Sorted by name so runs are stable.
    """
    folder = repo_dir / CI_SCRIPTS_DIR
    if not folder.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        found.append((path.stem, path.relative_to(repo_dir).as_posix()))
    return found

    """
    Return (job_name, checkout-relative path) for each script in tests/regression/ci.

    Skips underscore files such as __init__.py. Sorted by name so runs are stable.
    """
    folder = repo_dir / CI_SCRIPTS_DIR
    if not folder.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        found.append((path.stem, path.relative_to(repo_dir).as_posix()))
    return found
