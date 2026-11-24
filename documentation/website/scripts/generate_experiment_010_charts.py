"""Create Experiment 010 comparison charts from completed run metadata."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

SITE = Path(__file__).resolve().parents[1]
# The run folder can be passed as the first argument when the runs live in another checkout.
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    SITE.parents[1] / "experiments" / "output" / "train_cifar10" / "runs" / "exp010_mcts_meta_g10_tiny")
PARAMETER_TITLES = {
    "MCTS_UCB1_C": "C", "MCTS_UCB1_USE_SQRT": "square root", "MCTS_PROPAGATE_ROLLOUT_VALUE": "propagate",
    "MCTS_ROLLOUT_DEPTH": "depth", "MCTS_ROLLOUT_EPOCHS": "rollout epochs", "MCTS_ROLLOUT_LR": "rollout LR",
    "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS": "min iterations", "simulation_time": "search time",
    "simulation_set_percent": "simulation set %",
}
OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DATASETS = ("cifar10",)
COLORS = {"cifar10": "#2563eb"}


def _label(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _load_rows(pass_number: int, parameter: str) -> list[dict[str, object]]:
    rows = []
    for path in sorted((RUNS / f"pass_{pass_number}" / parameter).glob("value_*/*/seed_*/run.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("dataset") in DATASETS:
            rows.append(row)
    return rows


def _candidate_order(rows: list[dict[str, object]]) -> list[object]:
    unique = list(dict.fromkeys(row["candidate_value"] for row in rows))
    if unique and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in unique):
        return sorted(unique, key=float)
    return sorted(unique, key=str)


def _chart(pass_number: int, parameter: str, rows: list[dict[str, object]]) -> None:
    candidates = _candidate_order(rows)
    if not candidates:
        return
    x = np.arange(len(candidates), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(candidates) * 1.35), 5.2), constrained_layout=True)
    for offset, dataset in enumerate(DATASETS):
        means, deviations = [], []
        for candidate in candidates:
            values = [
                float(row["final_validation_accuracy"])
                for row in rows
                if row["dataset"] == dataset and row["candidate_value"] == candidate
            ]
            means.append(float(np.mean(values)) if values else np.nan)
            deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        positions = x + (offset - 0.5) * width
        ax.bar(positions, means, width, yerr=deviations, capsize=3, color=COLORS[dataset], label=dataset.upper())
    ax.set_xticks(x, [_label(value) for value in candidates])
    ax.set_xlabel("Candidate value")
    ax.set_ylabel("Final validation accuracy")
    ax.set_title(f"Experiment 010 · Pass {pass_number} · {parameter}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(OUTPUT / f"010-pass-{pass_number}-{parameter.lower()}.png", dpi=160)
    plt.close(fig)


def _value_label(folder_value: str) -> str:
    """Folder names store numbers with 'p' for the decimal point, for example 10p0 or 0p28444."""
    try:
        return f"{float(folder_value.replace('p', '.')):.3g}"
    except ValueError:
        return folder_value


def _value_order(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def _training_history_chart(pass_number: int, parameter: str) -> None:
    """Plot train/validation histories from board metrics for the active sweep."""
    run_dirs = sorted((RUNS / f"pass_{pass_number}" / parameter).glob("value_*/*/seed_*"))
    records = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "board" / "metrics" / "training.json"
        board_path = run_dir / "board" / "main.json"
        if not metrics_path.is_file() or not board_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        board = json.loads(board_path.read_text(encoding="utf-8"))
        records.append({
            "candidate": _value_label(run_dir.parent.parent.name.removeprefix("value_")),
            "dataset": board["dataset"],
            "seed": int(run_dir.name.removeprefix("seed_")),
            "status": board["status"],
            "epochs": metrics["epochs"],
        })
    if not records:
        return
    candidates = sorted({row["candidate"] for row in records}, key=_value_order)
    present_datasets = [dataset for dataset in DATASETS if any(row["dataset"] == dataset for row in records)]
    panels = [(candidate, dataset) for candidate in candidates for dataset in present_datasets]
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows, columns,
        figsize=(4.2 * columns, 3.6 * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    seed_colors = {1: "C0", 2: "C1"}
    flat_axes = list(axes.flat)
    for panel_index, (candidate, dataset) in enumerate(panels):
        ax = flat_axes[panel_index]
        subset = [r for r in records if r["candidate"] == candidate and r["dataset"] == dataset]
        for item in sorted(subset, key=lambda r: r["seed"]):
            epochs = item["epochs"]
            x = [int(r["globalEpoch"]) + 1 for r in epochs]
            color = seed_colors.get(item["seed"], "C7")
            ax.plot(x, [float(r["trainAcc"]) * 100 for r in epochs], color=color, lw=1.35,
                    label=f"seed {item['seed']} train")
            ax.plot(x, [float(r["valAcc"]) * 100 for r in epochs], color=color, lw=1.25,
                    ls="--", label=f"seed {item['seed']} validation")
        if subset:
            n_epochs = max(len(item["epochs"]) for item in subset)
            for boundary in range(20, 201, 20):
                if boundary < n_epochs:
                    ax.axvline(boundary + 0.5, color="gray", alpha=0.5, lw=0.8, zorder=0)
            ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.set_title(f"{dataset.upper()} · {PARAMETER_TITLES.get(parameter, parameter)} = {candidate}")
        ax.set_xlabel("Global epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.22, axis="y")
        ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.9)
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Experiment 010 · Pass {pass_number} · Training histories by {parameter}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    chart_parameter = parameter.lower().removeprefix("mcts_").replace("_", "-")
    fig.savefig(OUTPUT / f"010-pass-{pass_number}-{chart_parameter}-training-histories-grid.svg")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": "010_mcts_meta", "passes": {}}
    state_path = RUNS / "experiment_state.json"
    if state_path.is_file():
        manifest["state"] = json.loads(state_path.read_text(encoding="utf-8"))
    for pass_number in (1, 2):
        pass_manifest = {}
        pass_dir = RUNS / f"pass_{pass_number}"
        if not pass_dir.is_dir():
            continue
        for parameter_dir in sorted(path for path in pass_dir.iterdir() if path.is_dir()):
            rows = _load_rows(pass_number, parameter_dir.name)
            if not rows:
                continue
            pass_manifest[parameter_dir.name] = rows
            _chart(pass_number, parameter_dir.name, rows)
            _training_history_chart(pass_number, parameter_dir.name)
        manifest["passes"][str(pass_number)] = pass_manifest
    (SITE / "data" / "experiments" / "experiment-010-mcts-meta.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote charts and snapshot from {RUNS}")


if __name__ == "__main__":
    main()
