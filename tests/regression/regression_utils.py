import argparse
import os
import shutil
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt

from growingnn.core.logger import logger

FOLDER_NAME = "testResults/regression"


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
        shutil.rmtree(FOLDER_NAME)
    os.makedirs(FOLDER_NAME)


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
