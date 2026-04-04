import argparse
import os
import shutil
from typing import Optional, Sequence

FOLDER_NAME = "testResults/regression"


def clear_regression_folder():
    if os.path.exists(FOLDER_NAME):
        shutil.rmtree(FOLDER_NAME)
    os.makedirs(FOLDER_NAME)


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
