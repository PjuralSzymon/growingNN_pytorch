"""Unit tests for the FastAPI trigger and dashboard endpoints."""

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from worker.app import JobRunner, create_app
from worker.auth import session_token

JOBS = {"mnist": {"script": "experiments/regression_ci/mnist.py", "dataset": "mnist"}}
BASELINES = {"mnist": {"mean_val_acc": 0.85, "max_params": 20000}}
RESULT_LINE = (
    "REGRESSION_CI_RESULT "
    '{"dataset": "mnist", "seeds": [100, 101], '
    '"val_acc": [0.9, 0.91], "param_count": [1000, 1100]}'
)


def _runner(tmp_path: Path, stdout: str = RESULT_LINE, returncode: int = 0) -> JobRunner:
    comments: list[dict[str, object]] = []

    def checkout_sha(work_dir, **_kwargs):
        repo_dir = work_dir / "repo"
        repo_dir.mkdir(parents=True)
        return repo_dir

    def run_dataset_script(_repo_dir, _script):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    def upsert_pr_comment(**kwargs):
        comments.append(kwargs)

    runner = JobRunner(
        workspaces_dir=tmp_path / "workspaces",
        runs_dir=tmp_path / "runs",
        jobs=JOBS,
        baselines=BASELINES,
        github_repo="owner/repo",
        github_token="token",
        mnist_cache_dir=tmp_path / "mnist-cache",
        checkout_sha=checkout_sha,
        run_dataset_script=run_dataset_script,
        upsert_pr_comment=upsert_pr_comment,
    )
    runner.comments = comments
    return runner


def _wait(runner: JobRunner, job_id: str) -> dict:
    for _ in range(80):
        record = runner.status(job_id)
        if record and record["state"] in {"done", "error"}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_start_jobs_requires_bearer_secret(tmp_path: Path, monkeypatch):
    """
    POST /v1/jobs should return 401 when the shared secret is missing or wrong.
    """

    # Arrange
    monkeypatch.setenv("CI_SHARED_SECRET", "secret")
    client = TestClient(create_app(_runner(tmp_path), web_dist=tmp_path / "missing-web"))

    # Act
    denied = client.post("/v1/jobs", json={"sha": "abc", "jobs": ["mnist"]})
    wrong = client.post(
        "/v1/jobs",
        json={"sha": "abc", "jobs": ["mnist"]},
        headers={"Authorization": "Bearer nope"},
    )

    # Assert
    assert denied.status_code == 401
    assert wrong.status_code == 401


def test_start_jobs_returns_202_and_comments_on_success(tmp_path: Path, monkeypatch):
    """
    POST /v1/jobs should accept the job and later comment MNIST results vs baseline.
    """

    # Arrange
    monkeypatch.setenv("CI_SHARED_SECRET", "secret")
    runner = _runner(tmp_path)
    client = TestClient(create_app(runner, web_dist=tmp_path / "missing-web"))

    # Act
    started = client.post(
        "/v1/jobs",
        json={"sha": "abc123", "pr": 7, "jobs": ["mnist"]},
        headers={"Authorization": "Bearer secret"},
    )
    job_id = started.json()["id"]
    finished = _wait(runner, job_id)

    # Assert
    assert started.status_code == 202
    assert finished["state"] == "done"
    assert len(runner.comments) == 1
    assert runner.comments[0]["pr"] == 7
    assert "better" in runner.comments[0]["body"]
    runs = list((tmp_path / "runs").glob("*.json"))
    assert len(runs) == 1


def test_start_jobs_rejects_unknown_job(tmp_path: Path, monkeypatch):
    """
    POST /v1/jobs should return 400 when the job name is not in the image map.
    """

    # Arrange
    monkeypatch.setenv("CI_SHARED_SECRET", "secret")
    client = TestClient(create_app(_runner(tmp_path), web_dist=tmp_path / "missing-web"))

    # Act
    response = client.post(
        "/v1/jobs",
        json={"sha": "abc", "jobs": ["imagenet"]},
        headers={"Authorization": "Bearer secret"},
    )

    # Assert
    assert response.status_code == 400
    assert "unknown job" in response.json()["error"]


def test_login_and_runs_require_dashboard_password(tmp_path: Path, monkeypatch):
    """
    /api/runs should stay closed until /api/login sets the dashboard cookie.
    """

    # Arrange
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pass")
    monkeypatch.setenv("CI_SHARED_SECRET", "secret")
    runner = _runner(tmp_path)
    client = TestClient(create_app(runner, web_dist=tmp_path / "missing-web"))

    # Act
    blocked = client.get("/api/runs")
    denied = client.post("/api/login", json={"password": "wrong"})
    allowed = client.post("/api/login", json={"password": "pass"})
    timeline = client.get("/api/runs")

    # Assert
    assert blocked.status_code == 401
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert timeline.status_code == 200
    assert timeline.json() == []
    assert client.cookies.get("regression_ci") == session_token("pass")
