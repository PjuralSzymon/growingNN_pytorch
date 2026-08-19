import argparse
import os
import shutil
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt

from growingnn.core.logger import logger

FOLDER_NAME = "testResults/regression"

ACTION_TYPE_COLUMNS = (
    "AddResLinearLayer",
    "AddResConvLayer",
    "AddSeqLinearLayer",
    "AddSeqConvLayer",
    "AddSeqDropoutLayer",
    "AddNeurons",
    "DelLayer",
    "DelNeurons",
)


def action_count_table_lines(
    counts: dict[str, int],
    *,
    title: str = "action counts",
    include_known_zeros: bool = True,
) -> list[str]:
    """ASCII table of action type counts, optionally including known types with zero."""
    merged = dict(counts)
    if include_known_zeros:
        merged = {name: counts.get(name, 0) for name in ACTION_TYPE_COLUMNS}
        for name, n in counts.items():
            merged[name] = n
        names = [n for n in ACTION_TYPE_COLUMNS if n in merged] + [
            n for n in sorted(merged) if n not in ACTION_TYPE_COLUMNS
        ]
    else:
        names = sorted(merged)
    col = max([len("action"), *(len(n) for n in names)], default=6)
    rows = [
        title,
        f"{'action':<{col}} | count",
        f"{'-' * col}-+------",
    ]
    for name in names:
        rows.append(f"{name:<{col}} | {merged[name]}")
    rows.append(f"{'total':<{col}} | {sum(merged.values())}")
    return rows


def log_action_count_table(
    counts: dict[str, int],
    *,
    title: str = "action counts",
    include_known_zeros: bool = True,
) -> list[str]:
    """Log one line per table row and return the same rows for a summary file."""
    rows = action_count_table_lines(
        counts, title=title, include_known_zeros=include_known_zeros
    )
    for row in rows:
        logger.info("%s", row)
    return rows


def regression_cifar_dir() -> str:
    """Prefer cached CIFAR under testResults, else experiments download."""
    for data_dir in (
        "testResults/regression_cache/cifar10",
        "experiments/output/train_from_repo/data",
    ):
        if os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py")):
            return data_dir
    return "testResults/regression_cache/cifar10"


def log_regression_action_error(
    gm,
    chosen,
    *,
    actions=None,
    idx: Optional[int] = None,
    action_type: Optional[str] = None,
    norms=None,
    parameter_amounts=None,
    **extra: Any,
) -> None:
    """Log FX graph and action context after a failed regression action execute."""
    logger.info("gm.graph: %s", gm.graph)
    if action_type is not None:
        logger.info("action type: %s", action_type)
    if actions is not None:
        logger.info("actions: %s", actions)
    if idx is not None:
        logger.info("idx: %s", idx)
        if actions is not None:
            logger.info("actions[idx]: %s", actions[idx])
    logger.info("chosen: %s", chosen)
    if norms is not None:
        logger.info("norms: %s", norms)
    if parameter_amounts is not None:
        logger.info("parameter_amounts: %s", parameter_amounts)
    for key, value in extra.items():
        logger.info("%s: %s", key, value)
    if action_type is not None:
        logger.exception("Error executing %s action %s", action_type, chosen)
    else:
        logger.exception("Error executing action %s", chosen)


def clear_regression_folder():
    logger.debug("Clearing regression output folder %s", FOLDER_NAME)
    if os.path.exists(FOLDER_NAME):
        shutil.rmtree(FOLDER_NAME, ignore_errors=True)
    os.makedirs(FOLDER_NAME, exist_ok=True)


def plot_norms_and_parameter_count(
    norms: Sequence[float],
    parameter_amounts: Sequence[int],
    *,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Twin-axis plot: ``norms`` vs step and param count (``parameter_amounts[1:]`` aligned to norms)."""
    if not norms:
        return
    steps = range(len(norms))
    fig, ax1 = plt.subplots()
    ax1.plot(steps, norms, color="C0")
    ax1.set_xlabel("step")
    ax1.set_ylabel("||Δout||", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(steps, parameter_amounts[1 : 1 + len(norms)], color="C1")
    ax2.set_ylabel("amount of params", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Saved norms/params plot: %s", save_path)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def parse_regression_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression harness CLI")
    parser.add_argument(
        "--save-output",
        "--save_output",
        choices=("true", "false"),
        default="false",
        help="Save FX graph PDFs under testResults/regression (default: false)",
    )
    ns = parser.parse_args(argv)
    ns.save_output = ns.save_output == "true"
    return ns
