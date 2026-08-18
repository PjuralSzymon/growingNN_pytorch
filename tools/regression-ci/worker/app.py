"""Always-on FastAPI worker: start jobs, serve the Angular timeline."""

from __future__ import annotations

import os
import queue
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from worker import auth, checkout, github, jobs as job_map, results, storage

MAX_QUEUE = 4
COOKIE_NAME = auth.COOKIE_NAME


class StartJobsBody(BaseModel):
    sha: str
    jobs: list[str] = Field(min_length=1)
    pr: int | None = None


class LoginBody(BaseModel):
    password: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


class JobRunner:
    """One-at-a-time queue that checkouts a SHA and runs dataset scripts."""

    def __init__(
        self,
        *,
        workspaces_dir: Path,
        runs_dir: Path,
        jobs: dict[str, dict[str, str]],
        baselines: dict[str, dict[str, float]],
        github_repo: str = "",
        github_token: str = "",
        mnist_cache_dir: Path | None = None,
        checkout_sha: Callable[..., Path] = checkout.checkout_sha,
        run_dataset_script: Callable[..., Any] = checkout.run_dataset_script,
        upsert_pr_comment: Callable[..., None] = github.upsert_pr_comment,
    ) -> None:
        self.workspaces_dir = workspaces_dir
        self.runs_dir = runs_dir
        self.jobs = jobs
        self.baselines = baselines
        self.github_repo = github_repo
        self.github_token = github_token
        self.mnist_cache_dir = mnist_cache_dir or Path(
            os.environ.get("MNIST_CACHE_DIR", "/data/mnist")
        )
        self._checkout_sha = checkout_sha
        self._run_dataset_script = run_dataset_script
        self._upsert_pr_comment = upsert_pr_comment
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._status: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, *, sha: str, pr: int | None, job_names: list[str]) -> str:
        unknown = [name for name in job_names if name not in self.jobs]
        if unknown:
            raise KeyError(",".join(unknown))
        if self._queue.qsize() >= MAX_QUEUE:
            raise RuntimeError("queue full")
        job_id = uuid.uuid4().hex[:12]
        item = {"id": job_id, "sha": sha, "pr": pr, "jobs": list(job_names)}
        with self._lock:
            self._status[job_id] = {"id": job_id, "state": "queued", **item}
        self._queue.put(item)
        return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._status.get(job_id)
            return None if record is None else dict(record)

    def _set_state(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._status[job_id].update(fields)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self._run_item(item)
            finally:
                self._queue.task_done()

    def _run_item(self, item: dict[str, Any]) -> None:
        job_id = item["id"]
        self._set_state(job_id, state="running")
        work_dir = self.workspaces_dir / job_id
        try:
            repo_dir = self._checkout_sha(
                work_dir,
                repo=self.github_repo,
                sha=item["sha"],
                token=self.github_token,
            )
            checkout.link_mnist_cache(repo_dir, self.mnist_cache_dir)
            for name in item["jobs"]:
                self._run_named_job(item, repo_dir, name)
            self._set_state(job_id, state="done")
        except Exception as exc:
            self._set_state(job_id, state="error", error=str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_named_job(self, item: dict[str, Any], repo_dir: Path, name: str) -> None:
        spec = self.jobs[name]
        dataset = spec["dataset"]
        finished_at = _utc_stamp()
        process = self._run_dataset_script(repo_dir, spec["script"])
        combined = f"{process.stdout or ''}\n{process.stderr or ''}"
        record: dict[str, Any] = {
            "job": name,
            "dataset": dataset,
            "sha": item["sha"],
            "pr": item["pr"],
            "finished_at": finished_at,
        }
        if process.returncode != 0:
            error = combined.strip() or f"exit {process.returncode}"
            record["ok"] = False
            record["error"] = error
            storage.save_run(self.runs_dir, record)
            self._comment(item["pr"], name, results.format_failure_comment(name, error))
            return
        try:
            metrics = results.parse_script_output(combined)
            baseline = self.baselines[dataset]
            comparison = results.compare_to_baseline(metrics, baseline)
        except Exception as exc:
            record["ok"] = False
            record["error"] = str(exc)
            storage.save_run(self.runs_dir, record)
            self._comment(item["pr"], name, results.format_failure_comment(name, str(exc)))
            return
        record.update(
            {
                "ok": True,
                "metrics": metrics,
                "comparison": comparison,
                "passed": comparison["passed"],
            }
        )
        storage.save_run(self.runs_dir, record)
        self._comment(
            item["pr"],
            name,
            results.format_comment(name, metrics, comparison, baseline),
        )

    def _comment(self, pr: int | None, job: str, body: str) -> None:
        if pr is None or not self.github_token or not self.github_repo:
            return
        self._upsert_pr_comment(
            repo=self.github_repo,
            pr=pr,
            job=job,
            body=body,
            token=self.github_token,
        )


def create_app(
    runner: JobRunner | None = None,
    *,
    web_dist: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app. Tests pass a runner with mocked git/GitHub."""
    runs_dir = Path(os.environ.get("RUNS_DIR", "/data/runs"))
    workspaces_dir = Path(os.environ.get("WORKSPACES_DIR", "/workspaces"))
    dist = web_dist if web_dist is not None else Path(os.environ.get("WEB_DIST", "/app/web/dist/browser"))
    if runner is None:
        runner = JobRunner(
            workspaces_dir=workspaces_dir,
            runs_dir=runs_dir,
            jobs=job_map.load_jobs(),
            baselines=job_map.load_baselines(),
            github_repo=os.environ.get("GITHUB_REPO", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            mnist_cache_dir=Path(os.environ.get("MNIST_CACHE_DIR", "/data/mnist")),
        )

    app = FastAPI()
    app.state.runner = runner
    app.state.runs_dir = runner.runs_dir
    app.state.web_dist = dist

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/jobs")
    def start_jobs(body: StartJobsBody, request: Request) -> JSONResponse:
        if not auth.bearer_matches(request.headers.get("authorization")):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            job_id = runner.enqueue(sha=body.sha, pr=body.pr, job_names=body.jobs)
        except KeyError as exc:
            return JSONResponse({"error": f"unknown job {exc}"}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"id": job_id, "state": "queued"}, status_code=202)

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> JSONResponse:
        if not auth.bearer_matches(request.headers.get("authorization")):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        record = runner.status(job_id)
        if record is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(record)

    @app.post("/api/login")
    def login(body: LoginBody) -> JSONResponse:
        password = auth.dashboard_password()
        if not password or body.password != password:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            COOKIE_NAME,
            auth.session_token(password),
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 14,
        )
        return response

    @app.get("/api/runs")
    def api_runs(request: Request) -> JSONResponse:
        if not auth.cookie_matches(request.cookies.get(COOKIE_NAME)):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse(storage.list_runs(runner.runs_dir))

    browser = dist
    if (browser / "index.html").is_file():
        assets = browser / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            candidate = browser / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(browser / "index.html")

    return app
