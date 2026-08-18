"""Checkout a commit SHA and run a dataset script from that tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def checkout_sha(
    work_dir: Path,
    *,
    repo: str,
    sha: str,
    token: str,
) -> Path:
    """Fetch and check out sha into work_dir/repo. Return that path."""
    repo_dir = work_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    remote = f"https://x-access-token:{token}@github.com/{repo}.git"
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", remote],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", sha],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    return repo_dir


def link_mnist_cache(repo_dir: Path, cache_dir: Path) -> None:
    """Point the checkout MNIST folder at a persistent volume so downloads survive jobs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = repo_dir / "experiments" / "data" / "mnist"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(cache_dir)
    except OSError:
        target.mkdir(parents=True, exist_ok=True)


def run_dataset_script(repo_dir: Path, script: str) -> subprocess.CompletedProcess[str]:
    """Run a checkout-relative Python script with the checkout on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, script],
        cwd=repo_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
