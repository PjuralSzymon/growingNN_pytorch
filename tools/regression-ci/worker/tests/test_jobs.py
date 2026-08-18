"""Unit tests for discovering scripts under tests/regression/ci."""

from pathlib import Path

from worker.jobs import CI_SCRIPTS_DIR, discover_ci_scripts


def test_discover_ci_scripts_runs_every_py_file_in_the_ci_folder(tmp_path: Path):
    """
    discover_ci_scripts should return every *.py in tests/regression/ci except underscore files.
    """

    # Arrange
    folder = tmp_path / CI_SCRIPTS_DIR
    folder.mkdir(parents=True)
    (folder / "mnist.py").write_text("# mnist\n", encoding="utf-8")
    (folder / "cifar.py").write_text("# cifar\n", encoding="utf-8")
    (folder / "__init__.py").write_text("", encoding="utf-8")

    # Act
    found = discover_ci_scripts(tmp_path)

    # Assert
    assert found == [
        ("cifar", "tests/regression/ci/cifar.py"),
        ("mnist", "tests/regression/ci/mnist.py"),
    ]


def test_discover_ci_scripts_returns_empty_when_folder_missing(tmp_path: Path):
    """
    discover_ci_scripts should return [] when the checkout has no CI folder yet.
    """

    # Arrange / Act
    found = discover_ci_scripts(tmp_path)

    # Assert
    assert found == []
