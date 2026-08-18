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
from pydantic import BaseModel, ConfigDict

from worker import auth, checkout, github, jobs as job_map, results, storage

MAX_QUEUE = 4
COOKIE_NAME = auth.COOKIE_NAME


class StartJobsBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sha: str
    pr: int | None = None


class LoginBody(BaseModel):
    password: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_sha(sha: str) -> str:
    return sha[:7] if sha else sha


def _error_text(exc: BaseException, token: str = "") -> str:
    parts = [str(exc)]
    output = getattr(exc, "output", None) or getattr(exc, "stderr", None)
    if output:
        parts.append(str(output))
    text = "\n".join(parts)[-4000:]
    if token:
        text = text.replace(token, "***")
    return text


class JobRunner:
    """One-at-a-time queue that checkouts a SHA and runs dataset scripts."""

    def __init__(
        self,
        *,
        workspaces_dir: Path,
        runs_dir: Path,
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

    def enqueue(self, *, sha: str, pr: int | None) -> str:
        if self._queue.qsize() >= MAX_QUEUE:
            raise RuntimeError("queue full")
        job_id = uuid.uuid4().hex[:12]
        now = _utc_stamp()
        record = {
            "id": job_id,
            "sha": sha,
            "pr": pr,
            "state": "queued",
            "step": "Waiting in queue",
            "queued_at": now,
            "updated_at": now,
            "results": [],
        }
        with self._lock:
            self._status[job_id] = record
        storage.save_job(self.runs_dir, record)
        self._queue.put({"id": job_id, "sha": sha, "pr": pr})
        return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._status.get(job_id)
            return None if record is None else dict(record)

    def _set_state(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._status[job_id].update(fields)
            self._status[job_id]["updated_at"] = _utc_stamp()
            record = dict(self._status[job_id])
        storage.save_job(self.runs_dir, record)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self._run_item(item)
            finally:
                self._queue.task_done()

    def _run_item(self, item: dict[str, Any]) -> None:
        job_id = item["id"]
        sha = str(item["sha"])
        log_path = storage.job_log_path(self.runs_dir, job_id)
        work_dir = self.workspaces_dir / job_id
        self._set_state(
            job_id,
            state="pulling",
            step=f"Fetching {_short_sha(sha)} from GitHub",
            pulling_at=_utc_stamp(),
        )
        try:
            repo_dir = self._checkout_sha(
                work_dir,
                repo=self.github_repo,
                sha=sha,
                token=self.github_token,
                log_path=log_path,
            )
            checkout.link_mnist_cache(repo_dir, self.mnist_cache_dir)
            scripts = job_map.discover_ci_scripts(repo_dir)
            if not scripts:
                error = "no scripts in tests/regression/ci"
                self._comment(
                    item["pr"], "ci", results.format_failure_comment("ci", error)
                )
                raise RuntimeError(error)
            for name, script in scripts:
                self._set_state(
                    job_id,
                    state="running",
                    step=f"Running {script}",
                    running_at=_utc_stamp(),
                )
                self._run_named_job(item, repo_dir, name, script, log_path)
            with self._lock:
                rows = list(self._status[job_id].get("results") or [])
            failed = [row for row in rows if not row.get("ok")]
            if failed:
                self._set_state(
                    job_id,
                    state="error",
                    step="Failed",
                    error=str(failed[-1].get("error") or "script failed"),
                    finished_at=_utc_stamp(),
                    passed=False,
                )
                return
            passed = all(row.get("passed", True) for row in rows)
            self._set_state(
                job_id,
                state="done",
                step="Done",
                finished_at=_utc_stamp(),
                passed=passed,
            )
        except Exception as exc:
            self._set_state(
                job_id,
                state="error",
                step="Failed",
                error=_error_text(exc, self.github_token),
                finished_at=_utc_stamp(),
                passed=False,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_named_job(
        self,
        item: dict[str, Any],
        repo_dir: Path,
        name: str,
        script: str,
        log_path: Path,
    ) -> None:
        finished_at = _utc_stamp()
        process = self._run_dataset_script(repo_dir, script, log_path=log_path)
        combined = f"{process.stdout or ''}\n{process.stderr or ''}"
        record: dict[str, Any] = {
            "job": name,
            "dataset": name,
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
            self._append_result(item["id"], record)
            return
        try:
            metrics = results.parse_script_output(combined)
            dataset = str(metrics.get("dataset", name))
            record["dataset"] = dataset
            baseline = self.baselines[dataset]
            comparison = results.compare_to_baseline(metrics, baseline)
        except Exception as exc:
            record["ok"] = False
            record["error"] = str(exc)
            storage.save_run(self.runs_dir, record)
            self._comment(item["pr"], name, results.format_failure_comment(name, str(exc)))
            self._append_result(item["id"], record)
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
        self._append_result(item["id"], record)

    def _append_result(self, job_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            rows = list(self._status[job_id].get("results") or [])
            rows.append(record)
            self._status[job_id]["results"] = rows
            snapshot = dict(self._status[job_id])
        storage.save_job(self.runs_dir, snapshot)

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


def _cookie_ok(request: Request) -> bool:
    return auth.cookie_matches(request.cookies.get(COOKIE_NAME))


def _public_job(record: dict[str, Any], *, log_tail: str | None = None) -> dict[str, Any]:
    payload = dict(record)
    if log_tail is not None:
        payload["log_tail"] = log_tail
    return payload


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
            job_id = runner.enqueue(sha=body.sha, pr=body.pr)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"id": job_id, "state": "queued"}, status_code=202)

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> JSONResponse:
        if not auth.bearer_matches(request.headers.get("authorization")):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        record = runner.status(job_id) or storage.load_job(runner.runs_dir, job_id)
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
        if not _cookie_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse(storage.list_runs(runner.runs_dir))

    @app.get("/api/jobs")
    def api_jobs(request: Request) -> JSONResponse:
        if not _cookie_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse(storage.list_jobs(runner.runs_dir))

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str, request: Request) -> JSONResponse:
        if not _cookie_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not job_id.isalnum():
            return JSONResponse({"error": "not found"}, status_code=404)
        record = runner.status(job_id) or storage.load_job(runner.runs_dir, job_id)
        if record is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            _public_job(
                record,
                log_tail=storage.read_log_tail(runner.runs_dir, job_id),
            )
        )

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
