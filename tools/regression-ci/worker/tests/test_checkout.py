"""Unit tests for git checkout of the requested commit SHA."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from worker.checkout import checkout_sha, link_mnist_cache


def test_checkout_sha_fetches_and_checks_out_the_requested_commit(tmp_path: Path):
    """
    checkout_sha should fetch the given SHA from GitHub and check out FETCH_HEAD.
    """

    # Arrange
    commands: list[list[str]] = []

    def fake_run(args, cwd=None, check=True, capture_output=True):
        commands.append(list(args))
        return SimpleNamespace(returncode=0)

    # Act
    with patch("worker.checkout.subprocess.run", fake_run):
        repo_dir = checkout_sha(
            tmp_path,
            repo="owner/repo",
            sha="abc123",
            token="tok",
        )

    # Assert
    assert repo_dir == tmp_path / "repo"
    assert ["git", "fetch", "--depth", "1", "origin", "abc123"] in commands
    assert ["git", "checkout", "FETCH_HEAD"] in commands
    remote = next(args for args in commands if args[:3] == ["git", "remote", "add"])
    assert "x-access-token:tok@github.com/owner/repo.git" in remote[-1]


def test_link_mnist_cache_symlinks_checkout_data_dir(tmp_path: Path):
    """
    link_mnist_cache should point experiments/data/mnist at the persistent cache folder.
    """

    # Arrange
    repo_dir = tmp_path / "repo"
    cache_dir = tmp_path / "cache"

    # Act
    link_mnist_cache(repo_dir, cache_dir)
    target = repo_dir / "experiments" / "data" / "mnist"

    # Assert
    assert target.exists()
    if target.is_symlink():
        assert target.resolve() == cache_dir.resolve()
