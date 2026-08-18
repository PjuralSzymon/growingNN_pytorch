"""Checkout a commit SHA and run a dataset script from that tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _redact(text: str, token: str) -> str:
    if not token or not text:
        return text
    return text.replace(token, "***")


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path | None,
    env: dict[str, str] | None = None,
    check: bool = True,
    token: str = "",
    display: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, optionally appending redacted stdout/stderr to log_path."""
    if log_path is None:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = _redact(display if display is not None else " ".join(command), token)
    chunks: list[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {label}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            safe = _redact(line, token)
            log.write(safe)
            log.flush()
            chunks.append(safe)
        code = process.wait()
    output = "".join(chunks)
    if check and code != 0:
        raise subprocess.CalledProcessError(code, command, output=output)
    return subprocess.CompletedProcess(command, code, stdout=output, stderr="")


def checkout_sha(
    work_dir: Path,
    *,
    repo: str,
    sha: str,
    token: str,
    log_path: Path | None = None,
) -> Path:
    """Fetch and check out sha into work_dir/repo. Return that path."""
    repo_dir = work_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    remote = f"https://x-access-token:{token}@github.com/{repo}.git"
    public_remote = f"https://github.com/{repo}.git"
    run_logged(["git", "init"], cwd=repo_dir, log_path=log_path, token=token)
    run_logged(
        ["git", "remote", "add", "origin", remote],
        cwd=repo_dir,
        log_path=log_path,
        token=token,
        display=f"git remote add origin {public_remote}",
    )
    run_logged(
        ["git", "fetch", "--depth", "1", "origin", sha],
        cwd=repo_dir,
        log_path=log_path,
        token=token,
    )
    run_logged(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=repo_dir,
        log_path=log_path,
        token=token,
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


def run_dataset_script(
    repo_dir: Path,
    script: str,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a checkout-relative Python script with the checkout on PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_dir)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    return run_logged(
        [sys.executable, script],
        cwd=repo_dir,
        env=env,
        log_path=log_path,
        check=False,
        display=f"python {script}",
    )
