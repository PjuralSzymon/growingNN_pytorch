"""GitHub issue-comment helpers for regression CI results."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from worker.results import COMMENT_MARKER_TEMPLATE

API_ROOT = "https://api.github.com"


def _request(
    method: str,
    url: str,
    token: str,
    body: dict[str, object] | None = None,
) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "growingnn-regression-ci",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {url} failed: {exc.code} {detail}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def upsert_pr_comment(
    *,
    repo: str,
    pr: int,
    job: str,
    body: str,
    token: str,
) -> None:
    """Create or replace the regression-ci comment for one job on a PR."""
    marker = COMMENT_MARKER_TEMPLATE.format(job=job)
    comments = _request(
        "GET",
        f"{API_ROOT}/repos/{repo}/issues/{pr}/comments?per_page=100",
        token,
    )
    if not isinstance(comments, list):
        raise RuntimeError("GitHub comments response was not a list")
    existing_id = next(
        (
            int(comment["id"])
            for comment in comments
            if marker in str(comment.get("body", ""))
        ),
        None,
    )
    if existing_id is None:
        _request(
            "POST",
            f"{API_ROOT}/repos/{repo}/issues/{pr}/comments",
            token,
            {"body": body},
        )
        return
    _request(
        "PATCH",
        f"{API_ROOT}/repos/{repo}/issues/comments/{existing_id}",
        token,
        {"body": body},
    )
