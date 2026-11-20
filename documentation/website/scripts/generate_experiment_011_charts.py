"""Generate Experiment 011 charts from history.pt files, or from the JSON snapshot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SITE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-011-simulation-sets.json"
DEFAULT_RUNS = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp011_simulation_sets"
)
ALGORITHMS = (
    "stratified_random",
    "el2n",
    "ccs",
    "moderate",
    "moderate_coverage",
)
LABELS = {
    "stratified_random": "Stratified random",
    "el2n": "EL2N",
    "ccs": "CCS",
    "moderate": "Moderate",
    "moderate_coverage": "Moderate coverage",
}
SEED_COLORS = {1: "#1d4ed8", 2: "#d97706"}


def _round_list(values: list[float]) -> list[float]:
    return [round(float(value), 6) for value in values]


def _from_history(path: Path) -> dict[str, object]:
    """Normalize one history.pt file into the snapshot row."""
    history = torch.load(path, map_location="cpu", weights_only=False)
    validation = [float(value) for value in history["val_acc"]]
    params = [int(value) for value in history["param_count"]]
    written = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    actions = [
        {"generation": int(history["action_generation"][index]), "name": str(name)}
        for index, name in enumerate(history["action_name"])
    ]
    return {
        "algorithm": str(history["algorithm"]),
        "seed": int(history["seed"]),
        "written": written,
        "train": _round_list(history["train_acc"]),
        "val": _round_list(validation),
        "test": _round_list(history["test_acc"]),
        "params_generation": [params[20 * index + 19] for index in range(len(history["test_acc"]))],
        "final_train": round(float(history["train_acc"][-1]), 6),
        "final_val": round(validation[-1], 6),
        "final_test": round(float(history["test_acc"][-1]), 6),
        "best_val": round(max(validation), 6),
        "params_end": params[-1],
        "actions": actions,
    }


def load_runs(run_dir: Path | None = None, snapshot: Path | None = None) -> dict[str, object]:
    """Load finished histories, falling back to the snapshot when raw output is absent."""
    run_dir = DEFAULT_RUNS if run_dir is None else run_dir
    snapshot = DEFAULT_SNAPSHOT if snapshot is None else snapshot
    histories = sorted(run_dir.glob("*/seed_*/history.pt")) if run_dir.is_dir() else []
    if histories:
        return {
            "experiment": "011_simulation_sets",
            "timeline_source": "filesystem mtime of history.pt",
            "runs": [_from_history(path) for path in histories],
        }
    if snapshot.is_file():
        return json.loads(snapshot.read_text(encoding="utf-8"))
    return {"experiment": "011_simulation_sets", "timeline_source": "filesystem mtime of history.pt", "runs": []}


def _rows(data: dict[str, object], algorithm: str) -> list[dict[str, object]]:
    return [row for row in data["runs"] if row["algorithm"] == algorithm]


def plot_final_accuracy(data: dict[str, object], out_dir: Path) -> None:
    """Mean final validation and test accuracy, with one marker per seed."""
    present = [name for name in ALGORITHMS if _rows(data, name)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=140)
    x = list(range(len(present)))
    width = 0.36
    for offset, key, label in ((-width / 2, "final_val", "Validation"), (width / 2, "final_test", "Official test")):
        means = []
        for name in present:
            values = [float(row[key]) * 100 for row in _rows(data, name)]
            means.append(sum(values) / len(values))
        ax.bar([position + offset for position in x], means, width, label=label, zorder=1)
    for row in data["runs"]:
        position = present.index(str(row["algorithm"]))
        color = SEED_COLORS.get(int(row["seed"]), "#64748b")
        ax.scatter(position - width / 2, float(row["final_val"]) * 100, color=color, s=28, zorder=3)
        ax.scatter(position + width / 2, float(row["final_test"]) * 100, color=color, marker="D", s=26, zorder=3)
    ax.set_xticks(x, [LABELS[name] for name in present])
    ax.set_ylabel("Final accuracy (%)")
    ax.set_xlabel("Simulation-set selector")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "011-final-accuracy.svg")
    plt.close(fig)


def plot_training_histories(data: dict[str, object], out_dir: Path) -> None:
    """One panel per selector, with training and validation accuracy for both seeds."""
    present = [name for name in ALGORITHMS if _rows(data, name)]
    if not present:
        return
    columns = len(present)
    fig, axes = plt.subplots(1, columns, figsize=(3.5 * columns, 4.2), dpi=140, sharey=True, squeeze=False)
    for axis, name in zip(axes[0], present):
        for row in sorted(_rows(data, name), key=lambda item: int(item["seed"])):
            epochs = list(range(1, len(row["train"]) + 1))
            color = SEED_COLORS.get(int(row["seed"]), "#64748b")
            axis.plot(epochs, [float(value) * 100 for value in row["train"]], color=color, lw=1.2, label=f"seed {row['seed']} train")
            axis.plot(epochs, [float(value) * 100 for value in row["val"]], color=color, lw=1.1, ls="--", label=f"seed {row['seed']} validation")
        for boundary in range(20, len(_rows(data, name)[0]["train"]), 20):
            axis.axvline(boundary, color="#94a3b8", lw=0.6, ls=":")
        axis.set_title(LABELS[name])
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=6, loc="lower right", framealpha=0.9)
    axes[0][0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "011-training-histories.svg")
    plt.close(fig)


def generate_charts(
    run_dir: Path | None = None,
    snapshot: Path | None = None,
    out_dir: Path | None = None,
    write_snapshot: bool = False,
) -> dict[str, object]:
    """Write the Experiment 011 figures and return the loaded rows."""
    out_dir = DEFAULT_OUTPUT if out_dir is None else out_dir
    snapshot = DEFAULT_SNAPSHOT if snapshot is None else snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_runs(run_dir, snapshot)
    source = DEFAULT_RUNS if run_dir is None else run_dir
    raw_present = source.is_dir() and any(source.glob("*/seed_*/history.pt"))
    if write_snapshot and raw_present and data["runs"]:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_final_accuracy(data, out_dir)
    plot_training_histories(data, out_dir)
    return data


def main() -> None:
    generate_charts(write_snapshot=True)


if __name__ == "__main__":
    main()
