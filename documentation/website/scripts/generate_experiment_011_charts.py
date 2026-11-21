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
ALGORITHM_COLORS = {
    "stratified_random": "#1d4ed8",
    "el2n": "#d97706",
    "ccs": "#059669",
    "moderate": "#7c3aed",
    "moderate_coverage": "#64748b",
}


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
        "dataset": str(history.get("dataset") or "cifar10"),
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
    histories = sorted(run_dir.glob("**/history.pt")) if run_dir.is_dir() else []
    if histories:
        return {
            "experiment": "011_simulation_sets",
            "timeline_source": "filesystem mtime of history.pt",
            "runs": [_from_history(path) for path in histories],
        }
    if snapshot.is_file():
        return json.loads(snapshot.read_text(encoding="utf-8"))
    return {"experiment": "011_simulation_sets", "timeline_source": "filesystem mtime of history.pt", "runs": []}


def _dataset_name(row: dict[str, object]) -> str:
    return str(row.get("dataset") or "cifar10")


def _datasets(data: dict[str, object]) -> list[str]:
    found = {_dataset_name(row) for row in data["runs"]}
    return [name for name in ("cifar10", "mnist", "cifar100") if name in found]


def _rows(data: dict[str, object], algorithm: str, dataset: str | None = None) -> list[dict[str, object]]:
    rows = [row for row in data["runs"] if row["algorithm"] == algorithm]
    if dataset is None:
        return rows
    return [row for row in rows if _dataset_name(row) == dataset]


DATASET_LABELS = {"cifar10": "CIFAR-10", "mnist": "MNIST", "cifar100": "CIFAR-100"}


def _draw_accuracy_bars(axis, data: dict[str, object], dataset: str | None) -> None:
    """Draw mean validation and test bars. dataset=None pools every finished run."""
    if dataset is None:
        present = [name for name in ALGORITHMS if _rows(data, name)]
        def rows_for(name: str) -> list[dict[str, object]]:
            return _rows(data, name)
    else:
        present = [name for name in ALGORITHMS if _rows(data, name, dataset)]
        def rows_for(name: str) -> list[dict[str, object]]:
            return _rows(data, name, dataset)
    if not present:
        return
    x = list(range(len(present)))
    width = 0.36
    for offset, key, label, color in (
        (-width / 2, "final_val", "Validation", "#1d4ed8"),
        (width / 2, "final_test", "Official test", "#d97706"),
    ):
        means = []
        for name in present:
            values = [float(row[key]) * 100 for row in rows_for(name)]
            means.append(sum(values) / len(values))
        axis.bar([position + offset for position in x], means, width, label=label, color=color, zorder=1)
    if dataset is not None:
        for row in _rows_for_dataset(data, dataset):
            position = present.index(str(row["algorithm"]))
            color = SEED_COLORS.get(int(row["seed"]), "#64748b")
            axis.scatter(position - width / 2, float(row["final_val"]) * 100, color=color, s=28, zorder=3)
            axis.scatter(position + width / 2, float(row["final_test"]) * 100, color=color, marker="D", s=26, zorder=3)
    axis.set_xticks(x, [LABELS[name] for name in present], rotation=15)
    axis.set_ylabel("Final accuracy (%)")
    axis.set_xlabel("Simulation-set selector")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()


def plot_pooled_final_accuracy(data: dict[str, object], out_dir: Path) -> None:
    """Range of final validation accuracy for every finished run of each selector."""
    present = [name for name in ALGORITHMS if _rows(data, name)]
    if not present:
        return
    present.sort(key=lambda name: sum(float(row["final_val"]) for row in _rows(data, name)) / len(_rows(data, name)))
    fig, axis = plt.subplots(figsize=(8.6, 4.8), dpi=140)
    dataset_marker = {"cifar10": "o", "mnist": "D"}
    for position, name in enumerate(present):
        rows = _rows(data, name)
        values = [float(row["final_val"]) * 100 for row in rows]
        axis.hlines(position, min(values), max(values), color="#94a3b8", lw=2, zorder=1)
        axis.scatter([sum(values) / len(values)], [position], color="#111827", s=36, zorder=3, label="Mean" if position == 0 else None)
        for row, value in zip(rows, values):
            dataset = _dataset_name(row)
            axis.scatter(
                [value],
                [position],
                marker=dataset_marker.get(dataset, "o"),
                color=SEED_COLORS.get(int(row["seed"]), "#64748b"),
                s=42,
                zorder=4,
                label=None,
            )
    axis.scatter([], [], marker="o", color="#64748b", s=42, label="CIFAR-10 run")
    axis.scatter([], [], marker="D", color="#64748b", s=42, label="MNIST run")
    axis.set_yticks(list(range(len(present))), [LABELS[name] for name in present])
    axis.set_xlabel("Final validation accuracy (%)")
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "011-final-accuracy.svg")
    plt.close(fig)


def plot_dataset_final_accuracy(data: dict[str, object], out_dir: Path) -> None:
    """Write one final-accuracy figure for each dataset that has finished runs."""
    for dataset in _datasets(data):
        fig, axis = plt.subplots(figsize=(8.4, 4.8), dpi=140)
        _draw_accuracy_bars(axis, data, dataset)
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
        fig.tight_layout()
        fig.savefig(out_dir / f"011-final-accuracy-{dataset}.svg")
        plt.close(fig)


def _rows_for_dataset(data: dict[str, object], dataset: str) -> list[dict[str, object]]:
    return [row for row in data["runs"] if _dataset_name(row) == dataset]


def plot_training_histories(data: dict[str, object], out_dir: Path) -> None:
    """One panel per selector and dataset, with training and validation accuracy."""
    datasets = _datasets(data)
    if not datasets:
        return
    columns = max(len([name for name in ALGORITHMS if _rows(data, name, dataset)]) for dataset in datasets)
    fig, axes = plt.subplots(
        len(datasets),
        columns,
        figsize=(3.4 * columns, 3.8 * len(datasets)),
        dpi=140,
        sharey=True,
        squeeze=False,
    )
    for row_index, dataset in enumerate(datasets):
        present = [name for name in ALGORITHMS if _rows(data, name, dataset)]
        for column, name in enumerate(present):
            axis = axes[row_index][column]
            rows = sorted(_rows(data, name, dataset), key=lambda item: int(item["seed"]))
            for row in rows:
                epochs = list(range(1, len(row["train"]) + 1))
                color = SEED_COLORS.get(int(row["seed"]), "#64748b")
                marker = "o" if int(row["seed"]) == 2 else None
                axis.plot(
                    epochs,
                    [float(value) * 100 for value in row["train"]],
                    color=color,
                    lw=1.15,
                    marker=marker,
                    markevery=10,
                    markersize=3.5,
                    label=f"seed {row['seed']} train",
                )
                axis.plot(
                    epochs,
                    [float(value) * 100 for value in row["val"]],
                    color=color,
                    lw=1.05,
                    ls="--",
                    marker=marker,
                    markevery=10,
                    markersize=3.5,
                    label=f"seed {row['seed']} validation",
                )
            for boundary in range(20, len(rows[0]["train"]), 20):
                axis.axvline(boundary, color="#94a3b8", lw=0.6, ls=":")
            axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} · {LABELS[name]}")
            axis.set_xlabel("Epoch")
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.22)
            axis.legend(fontsize=6, loc="lower right", framealpha=0.9)
        for column in range(len(present), columns):
            axes[row_index][column].axis("off")
        axes[row_index][0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "011-training-histories.svg")
    plt.close(fig)


def _mean_curve(rows: list[dict[str, object]], key: str) -> list[float]:
    """Average one accuracy series across the finished seeds, epoch by epoch."""
    series = [row[key] for row in rows]
    length = min(len(values) for values in series)
    means = []
    for index in range(length):
        values = [float(item[index]) for item in series]
        means.append(sum(values) / len(values))
    return means


def plot_mean_curves(data: dict[str, object], out_dir: Path, key: str, filename: str, ylabel: str) -> None:
    """One mean line per selector, averaging finished CIFAR-10 and MNIST runs together."""
    present = [name for name in ALGORITHMS if _rows(data, name)]
    if not present:
        return
    fig, axis = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    for name in present:
        rows = _rows(data, name)
        means = _mean_curve(rows, key)
        epochs = list(range(1, len(means) + 1))
        axis.plot(
            epochs,
            [value * 100 for value in means],
            color=ALGORITHM_COLORS[name],
            lw=1.8,
            label=f"{LABELS[name]} ({len(rows)} run{'s' if len(rows) != 1 else ''})",
        )
    length = min(len(row[key]) for name in present for row in _rows(data, name))
    for boundary in range(20, length, 20):
        axis.axvline(boundary, color="#94a3b8", lw=0.6, ls=":")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
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
    raw_present = source.is_dir() and any(source.glob("**/history.pt"))
    if write_snapshot and raw_present and data["runs"]:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_pooled_final_accuracy(data, out_dir)
    plot_dataset_final_accuracy(data, out_dir)
    plot_training_histories(data, out_dir)
    plot_mean_curves(data, out_dir, "train", "011-mean-training-accuracy.svg", "Mean training accuracy (%)")
    plot_mean_curves(data, out_dir, "val", "011-mean-validation-accuracy.svg", "Mean validation accuracy (%)")
    return data


def main() -> None:
    generate_charts(write_snapshot=True)


if __name__ == "__main__":
    main()
