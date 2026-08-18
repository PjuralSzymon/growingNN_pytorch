"""Unit tests for GitHub comment create-or-replace selection."""

from unittest.mock import patch

from worker.github import upsert_pr_comment


def test_upsert_pr_comment_posts_when_no_marker_exists():
    """
    upsert_pr_comment should POST a new comment when the job marker is absent.
    """

    # Arrange
    calls: list[tuple[str, str]] = []

    def fake_request(method, url, token, body=None):
        calls.append((method, url))
        if method == "GET":
            return [{"id": 1, "body": "unrelated"}]
        return {"id": 2}

    # Act
    with patch("worker.github._request", fake_request):
        upsert_pr_comment(
            repo="o/r",
            pr=3,
            job="mnist",
            body="<!-- regression-ci:mnist -->\nhi",
            token="t",
        )

    # Assert
    assert calls[0][0] == "GET"
    assert calls[1][0] == "POST"
    assert calls[1][1].endswith("/issues/3/comments")


def test_upsert_pr_comment_patches_existing_job_marker():
    """
    upsert_pr_comment should PATCH the existing comment that has the job marker.
    """

    # Arrange
    calls: list[tuple[str, str]] = []

    def fake_request(method, url, token, body=None):
        calls.append((method, url))
        if method == "GET":
            return [{"id": 9, "body": "<!-- regression-ci:mnist -->\nold"}]
        return {"id": 9}

    # Act
    with patch("worker.github._request", fake_request):
        upsert_pr_comment(
            repo="o/r",
            pr=3,
            job="mnist",
            body="<!-- regression-ci:mnist -->\nnew",
            token="t",
        )

    # Assert
    assert calls[1][0] == "PATCH"
    assert calls[1][1].endswith("/issues/comments/9")
