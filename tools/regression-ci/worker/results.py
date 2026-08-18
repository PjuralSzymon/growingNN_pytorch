"""Parse dataset-script output and compare it to hardcoded image baselines."""

from __future__ import annotations

import json
from typing import Any

RESULT_PREFIX = "REGRESSION_CI_RESULT "
COMMENT_MARKER_TEMPLATE = "<!-- regression-ci:{job} -->"


def parse_script_output(text: str) -> dict[str, Any]:
    """Read the last REGRESSION_CI_RESULT line from a dataset script."""
    for line in reversed(text.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise ValueError("dataset script printed no REGRESSION_CI_RESULT line")


def compare_to_baseline(
    metrics: dict[str, Any],
    baseline: dict[str, float],
) -> dict[str, Any]:
    """Compare mean accuracy and per-seed param counts to the image baseline."""
    val_acc = [float(value) for value in metrics["val_acc"]]
    param_count = [int(value) for value in metrics["param_count"]]
    mean_val_acc = sum(val_acc) / len(val_acc)
    mean_param_count = sum(param_count) / len(param_count)
    accuracy_better = mean_val_acc >= float(baseline["mean_val_acc"])
    params_within_cap = all(count <= int(baseline["max_params"]) for count in param_count)
    return {
        "mean_val_acc": mean_val_acc,
        "mean_param_count": mean_param_count,
        "accuracy": "better" if accuracy_better else "worse",
        "params": "within cap" if params_within_cap else "over cap",
        "passed": accuracy_better and params_within_cap,
    }


def format_comment(
    job: str,
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    baseline: dict[str, float],
) -> str:
    """Build the PR comment body for one finished dataset job."""
    marker = COMMENT_MARKER_TEMPLATE.format(job=job)
    seeds = metrics["seeds"]
    lines = [
        marker,
        f"Regression CI — `{job}` on `{metrics['dataset']}`",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        (
            f"Mean val acc: {comparison['mean_val_acc']:.2%} "
            f"(baseline {float(baseline['mean_val_acc']):.2%}) — {comparison['accuracy']}"
        ),
        (
            f"Mean params: {comparison['mean_param_count']:.0f} "
            f"(cap {int(baseline['max_params'])}) — {comparison['params']}"
        ),
        "",
    ]
    for seed, acc, params in zip(seeds, metrics["val_acc"], metrics["param_count"]):
        lines.append(f"- seed {seed}: val acc {float(acc):.2%}, params {int(params)}")
    return "\n".join(lines)


def format_failure_comment(job: str, error: str) -> str:
    """Build the PR comment body when the dataset script fails."""
    marker = COMMENT_MARKER_TEMPLATE.format(job=job)
    return f"{marker}\nRegression CI — `{job}` failed.\n\n```\n{error}\n```"
