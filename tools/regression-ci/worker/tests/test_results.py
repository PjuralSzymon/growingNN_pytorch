"""Unit tests for script-output parsing, baseline comparison, and PR comments."""

from worker.results import (
    compare_to_baseline,
    format_comment,
    format_failure_comment,
    parse_script_output,
)

BASELINE = {"mean_val_acc": 0.85, "max_params": 20000}


def test_parse_script_output_uses_last_result_line():
    """
    parse_script_output should ignore logs and keep the last REGRESSION_CI_RESULT line.
    """

    # Arrange
    text = "\n".join(
        [
            "loading mnist",
            'REGRESSION_CI_RESULT {"dataset": "mnist", "seeds": [100]}',
            'REGRESSION_CI_RESULT {"dataset": "mnist", "seeds": [100, 101]}',
        ]
    )

    # Act
    payload = parse_script_output(text)

    # Assert
    assert payload == {"dataset": "mnist", "seeds": [100, 101]}


def test_parse_script_output_raises_when_prefix_missing():
    """
    parse_script_output should fail when the dataset script prints no result line.
    """

    # Arrange / Act / Assert
    try:
        parse_script_output("no metrics here")
    except ValueError as exc:
        assert "REGRESSION_CI_RESULT" in str(exc)
    else:
        raise AssertionError("missing result line should raise ValueError")


def test_compare_to_baseline_marks_accuracy_better_and_params_within_cap():
    """
    Mean val acc at or above 85% and params under the cap should pass as better.
    """

    # Arrange
    metrics = {
        "val_acc": [0.879, 0.921],
        "param_count": [1200, 1300],
    }

    # Act
    comparison = compare_to_baseline(metrics, BASELINE)

    # Assert
    assert comparison["accuracy"] == "better"
    assert comparison["params"] == "within cap"
    assert comparison["passed"] is True
    assert comparison["mean_val_acc"] == 0.9


def test_compare_to_baseline_marks_accuracy_worse_and_params_over_cap():
    """
    Mean val acc below baseline or a seed over max_params should fail the gate.
    """

    # Arrange
    metrics = {
        "val_acc": [0.50, 0.51],
        "param_count": [100, 30000],
    }

    # Act
    comparison = compare_to_baseline(metrics, BASELINE)

    # Assert
    assert comparison["accuracy"] == "worse"
    assert comparison["params"] == "over cap"
    assert comparison["passed"] is False


def test_format_comment_includes_seeds_dataset_and_baseline_verdict():
    """
    The PR comment should name the job, seeds, means, and better/worse vs baseline.
    """

    # Arrange
    metrics = {
        "dataset": "mnist",
        "seeds": [100, 101],
        "val_acc": [0.879, 0.921],
        "param_count": [1200, 1300],
    }
    comparison = compare_to_baseline(metrics, BASELINE)

    # Act
    body = format_comment("mnist", metrics, comparison, BASELINE)

    # Assert
    assert "<!-- regression-ci:mnist -->" in body
    assert "`mnist`" in body
    assert "Seeds: 100, 101" in body
    assert "better" in body
    assert "within cap" in body
    assert "seed 100" in body


def test_format_failure_comment_includes_error_and_job_marker():
    """
    A crashed dataset script should still post a marked failure comment.
    """

    # Arrange / Act
    body = format_failure_comment("mnist", "boom")

    # Assert
    assert "<!-- regression-ci:mnist -->" in body
    assert "failed" in body
    assert "boom" in body
