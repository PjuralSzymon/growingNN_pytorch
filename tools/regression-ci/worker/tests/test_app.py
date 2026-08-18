"""Unit tests for the FastAPI trigger and dashboard endpoints."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from worker.app import JobRunner, create_app
from worker.auth import session_token
from worker import storage

BASELINES = {"mnist": {"mean_val_acc": 0.85, "max_params": 20000}}
RESULT_LINE = (
    "REGRESSION_CI_RESULT "
    '{"dataset": "mnist", "seeds": [100, 101], '
    '"val_acc": [0.9, 0.91], "param_count": [1000, 1100]}'
)


def _write_ci_script(repo_dir: Path, name: str = "mnist") -> None:
    script = repo_dir / "tests" / "regression" / "ci" / f"{name}.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# regression ci script\n", encoding="utf-8")


def _runner(tmp_path: Path, stdout: str = RESULT_LINE, returncode: int = 0) -> JobRunner:
    comments: list[dict[str, object]] = []
    ran: list[str] = []

    def checkout_sha(work_dir, **_kwargs):
        repo_dir = work_dir / "repo"
        _write_ci_script(repo_dir)
        return repo_dir

    def run_dataset_script(_repo_dir, script, **_kwargs):
        ran.append(script)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    def upsert_pr_comment(**kwargs):
        comments.append(kwargs)

    runner = JobRunner(
        workspaces_dir=tmp_path / "workspaces",
        runs_dir=tmp_path / "runs",
        baselines=BASELINES,
        github_repo="owner/repo",
        github_token="token",
        mnist_cache_dir=tmp_path / "mnist-cache",
        checkout_sha=checkout_sha,
        run_dataset_script=run_dataset_script,
        upsert_pr_comment=upsert_pr_comment,
    )
    runner.comments = comments
    runner.ran = ran
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
    denied = client.post("/v1/jobs", json={"sha": "abc"})
    wrong = client.post(
        "/v1/jobs",
        json={"sha": "abc"},
        headers={"Authorization": "Bearer nope"},
    )

    # Assert
    assert denied.status_code == 401
    assert wrong.status_code == 401


def test_start_jobs_returns_202_and_runs_scripts_from_ci_folder(tmp_path: Path, monkeypatch):
    """
    POST /v1/jobs should accept the SHA and later run every script in tests/regression/ci.
    """

    # Arrange
    monkeypatch.setenv("CI_SHARED_SECRET", "secret")
    runner = _runner(tmp_path)
    client = TestClient(create_app(runner, web_dist=tmp_path / "missing-web"))

    # Act
    started = client.post(
        "/v1/jobs",
        json={"sha": "abc123", "pr": 7},
        headers={"Authorization": "Bearer secret"},
    )
    job_id = started.json()["id"]
    finished = _wait(runner, job_id)

    # Assert
    assert started.status_code == 202
    assert finished["state"] == "done"
    assert runner.ran == ["tests/regression/ci/mnist.py"]
    assert len(runner.comments) == 1
    assert runner.comments[0]["pr"] == 7
    assert "better" in runner.comments[0]["body"]
    runs = list((tmp_path / "runs").glob("*.json"))
    assert len(runs) == 1


def test_start_jobs_errors_when_ci_folder_has_no_scripts(tmp_path: Path, monkeypatch):
    """
    A checkout with no tests/regression/ci scripts should finish in the error state.
    """

    # Arrange
    monkeypatch.setenv("CI_SHARED_SECRET", "secret")

    def checkout_sha(work_dir, **_kwargs):
        repo_dir = work_dir / "repo"
        repo_dir.mkdir(parents=True)
        return repo_dir

    runner = _runner(tmp_path)
    runner._checkout_sha = checkout_sha
    client = TestClient(create_app(runner, web_dist=tmp_path / "missing-web"))

    # Act
    started = client.post(
        "/v1/jobs",
        json={"sha": "abc", "pr": 1},
        headers={"Authorization": "Bearer secret"},
    )
    finished = _wait(runner, started.json()["id"])

    # Assert
    assert started.status_code == 202
    assert finished["state"] == "error"
    assert "tests/regression/ci" in finished["error"]


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


def test_enqueue_persists_job_before_checkout_finishes(tmp_path: Path):
    """
    enqueue should write status.json so the dashboard can list a job before work finishes.
    """

    # Arrange
    release = threading.Event()

    def checkout_sha(work_dir, **_kwargs):
        release.wait(timeout=5)
        repo_dir = work_dir / "repo"
        _write_ci_script(repo_dir)
        return repo_dir

    runner = _runner(tmp_path)
    runner._checkout_sha = checkout_sha

    # Act
    try:
        job_id = runner.enqueue(sha="abc123", pr=9)
        record = storage.load_job(runner.runs_dir, job_id)
    finally:
        release.set()
        _wait(runner, job_id)

    # Assert
    assert record is not None
    assert record["sha"] == "abc123"
    assert record["pr"] == 9
    assert record["state"] in {"queued", "pulling"}


def test_job_enters_pulling_before_checkout_returns(tmp_path: Path):
    """
    The job should be in the pulling state while git checkout is still running.
    """

    # Arrange
    entered = threading.Event()
    release = threading.Event()

    def checkout_sha(work_dir, **_kwargs):
        entered.set()
        release.wait(timeout=5)
        repo_dir = work_dir / "repo"
        _write_ci_script(repo_dir)
        return repo_dir

    runner = _runner(tmp_path)
    runner._checkout_sha = checkout_sha

    # Act
    job_id = runner.enqueue(sha="deadbeef", pr=3)
    try:
        assert entered.wait(timeout=2)
        record = runner.status(job_id)
    finally:
        release.set()
        _wait(runner, job_id)

    # Assert
    assert record is not None
    assert record["state"] == "pulling"
    assert "Fetch" in record["step"]


def test_job_enters_running_while_dataset_script_executes(tmp_path: Path):
    """
    The job should be in the running state while a dataset script is executing.
    """

    # Arrange
    entered = threading.Event()
    release = threading.Event()

    def run_dataset_script(_repo_dir, script, **_kwargs):
        entered.set()
        release.wait(timeout=5)
        return SimpleNamespace(returncode=0, stdout=RESULT_LINE, stderr="")

    runner = _runner(tmp_path)
    runner._run_dataset_script = run_dataset_script

    # Act
    job_id = runner.enqueue(sha="abc", pr=4)
    try:
        assert entered.wait(timeout=2)
        record = runner.status(job_id)
    finally:
        release.set()
        _wait(runner, job_id)

    # Assert
    assert record is not None
    assert record["state"] == "running"
    assert "mnist.py" in record["step"]


def test_api_jobs_requires_dashboard_cookie(tmp_path: Path, monkeypatch):
    """
    GET /api/jobs should return 401 until the dashboard cookie is set.
    """

    # Arrange
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pass")
    client = TestClient(create_app(_runner(tmp_path), web_dist=tmp_path / "missing-web"))

    # Act
    blocked = client.get("/api/jobs")
    client.post("/api/login", json={"password": "pass"})
    allowed = client.get("/api/jobs")

    # Assert
    assert blocked.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json() == []


def test_api_jobs_lists_checkout_failure(tmp_path: Path, monkeypatch):
    """
    A git checkout failure should stay visible on GET /api/jobs as state error.
    """

    # Arrange
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pass")

    def checkout_sha(_work_dir, **_kwargs):
        raise RuntimeError("git fetch failed")

    runner = _runner(tmp_path)
    runner._checkout_sha = checkout_sha
    client = TestClient(create_app(runner, web_dist=tmp_path / "missing-web"))
    client.post("/api/login", json={"password": "pass"})

    # Act
    job_id = runner.enqueue(sha="missing", pr=5)
    finished = _wait(runner, job_id)
    listed = client.get("/api/jobs")

    # Assert
    assert finished["state"] == "error"
    assert listed.status_code == 200
    rows = listed.json()
    assert rows[0]["id"] == job_id
    assert rows[0]["state"] == "error"
    assert "git fetch failed" in rows[0]["error"]


def test_api_job_returns_log_tail(tmp_path: Path, monkeypatch):
    """
    GET /api/jobs/{id} should include the last lines of the job log.
    """

    # Arrange
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pass")
    runner = _runner(tmp_path)
    client = TestClient(create_app(runner, web_dist=tmp_path / "missing-web"))
    client.post("/api/login", json={"password": "pass"})
    job_id = "abc123def456"
    storage.save_job(
        runner.runs_dir,
        {
            "id": job_id,
            "sha": "deadbeef",
            "pr": 8,
            "state": "pulling",
            "step": "Fetching deadbee from GitHub",
            "queued_at": "2026-08-18T12:00:00Z",
            "updated_at": "2026-08-18T12:00:01Z",
        },
    )
    storage.job_log_path(runner.runs_dir, job_id).write_text(
        "$ git fetch\nremote: Counting objects\n", encoding="utf-8"
    )

    # Act
    response = client.get(f"/api/jobs/{job_id}")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "pulling"
    assert "Counting objects" in body["log_tail"]
