"""Unit tests for job-name to checkout script path lookup."""

from pathlib import Path

from worker.jobs import load_jobs, script_path_for_job


def test_script_path_for_job_returns_mnist_script():
    """
    The image job map should send mnist to experiments/regression_ci/mnist.py.
    """

    # Arrange
    jobs = load_jobs()

    # Act
    path = script_path_for_job("mnist", jobs)

    # Assert
    assert path == "experiments/regression_ci/mnist.py"
    assert jobs["mnist"]["dataset"] == "mnist"


def test_script_path_for_job_rejects_unknown_name():
    """
    script_path_for_job should raise KeyError for a job that is not registered.
    """

    # Arrange
    jobs = {"mnist": {"script": "experiments/regression_ci/mnist.py", "dataset": "mnist"}}

    # Act / Assert
    try:
        script_path_for_job("cifar", jobs)
    except KeyError as exc:
        assert "cifar" in str(exc)
    else:
        raise AssertionError("unknown job should raise KeyError")


def test_load_jobs_reads_repo_jobs_json():
    """
    load_jobs should read jobs.json from the regression-ci folder.
    """

    # Arrange / Act
    jobs = load_jobs()

    # Assert
    assert Path(jobs["mnist"]["script"]).as_posix() == "experiments/regression_ci/mnist.py"
