import argparse
import os
import shutil
from typing import Optional, Sequence

import matplotlib.pyplot as plt

from growingnn.core.logger import logger

FOLDER_NAME = "testResults/regression"


def clear_regression_folder():
    logger.debug("Clearing regression output folder %s", FOLDER_NAME)
    if os.path.exists(FOLDER_NAME):
        shutil.rmtree(FOLDER_NAME)
    os.makedirs(FOLDER_NAME)


def plot_norms_and_parameter_count(
    norms: Sequence[float],
    parameter_amounts: Sequence[int],
) -> None:
    """Twin-axis plot: ``norms`` vs step, and param count (``parameter_amounts[1:]`` aligned to norms)."""
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
    plt.show()


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
