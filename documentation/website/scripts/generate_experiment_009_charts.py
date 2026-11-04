"""Generate Experiment 009 charts from finished history.pt files, or from the JSON snapshot."""

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
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-009-best-reference.json"
RUN_DIR = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp009_best_reference"
)
SMALL_RUN_DIR = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp009_small_20"
)
SMALL_SNAPSHOT = SITE / "data" / "experiments" / "experiment-009-small-20.json"
DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
DATASET_LABEL = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "tiny_imagenet": "Tiny ImageNet",
}
SEED_COLORS = {1: "#1d4ed8", 2: "#d97706"}
REFERENCE_TEST = {
    "cifar10": 91.21,
    "cifar100": None,
    "mnist": None,
    "tiny_imagenet": None,
}


def _load_history(path: Path) -> dict[str, object]:
    """Read one history.pt into the lists the charts and the page use."""
    history = torch.load(path, map_location="cpu", weights_only=False)
    test = [float(value) for value in history["test_acc"]]
    params = [int(value) for value in history["param_count"]]
    actions = []
    for index, name in enumerate(history["action_name"]):
        actions.append(
            {
                "generation": int(history["action_generation"][index]),
                "name": str(name),
                "val_before": float(history["action_val_acc_before"][index]),
                "val_after": float(history["action_val_acc_after"][index]),
            }
        )
    written = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "dataset": str(history["dataset"]),
        "seed": int(history["seed"]),
        "written": written,
        "train": [float(value) for value in history["train_acc"]],
        "val": [float(value) for value in history["val_acc"]],
        "test": test,
        "test_epoch": [10 * (index + 1) for index in range(len(test))],
        "params_epoch": params,
        "params_generation": [params[10 * index + 9] for index in range(len(test))],
        "final_train": float(history["train_acc"][-1]),
        "final_val": float(history["val_acc"][-1]),
        "final_test": test[-1],
        "params_start": params[0],
        "params_end": params[-1],
        "actions": actions,
    }


def load_measurement(run_dir: Path) -> dict[str, object]:
    """Load every finished dataset seed. ImageNet-100 and ImageNet are omitted until history.pt exists."""
    rows = []
    for dataset in DATASETS:
        for path in sorted((run_dir / dataset).glob("seed_*/history.pt")):
            rows.append(_load_history(path))
    return {"runs": rows}


def _rows(data: dict[str, object], dataset: str) -> list[dict[str, object]]:
    return [row for row in data["runs"] if row["dataset"] == dataset]


def _present(data: dict[str, object]) -> list[str]:
    """Datasets that have at least one history, in the small-to-large order."""
    return [dataset for dataset in DATASETS if _rows(data, dataset)]


def _panels(count: int, wide: tuple[float, float], grid: tuple[float, float]):
    """One row when two datasets are finished, otherwise the two-by-two grid."""
    if count <= 2:
        fig, axes = plt.subplots(1, count, figsize=(wide[0], wide[1]), dpi=140, sharex=True)
        return fig, [axes] if count == 1 else list(axes)
    fig, axes = plt.subplots(2, 2, figsize=(grid[0], grid[1]), dpi=140, sharex=True)
    return fig, list(axes.ravel())


def plot_final_test(data: dict[str, object], out_dir: Path, filename: str = "009-final-official-test.png") -> None:
    """Grouped bars of the last official test for each finished dataset."""
    present = _present(data)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=140)
    width = 0.36
    for offset, seed in enumerate((1, 2)):
        heights = []
        positions = []
        for index, dataset in enumerate(present):
            match = next((row for row in _rows(data, dataset) if int(row["seed"]) == seed), None)
            if match is None:
                continue
            heights.append(100.0 * float(match["final_test"]))
            positions.append(index + (offset - 0.5) * width)
        if heights:
            ax.bar(positions, heights, width=width, color=SEED_COLORS[seed], label=f"Seed {seed}")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([DATASET_LABEL[name] for name in present])
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color="#e2e8f0")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_train_histories(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-train-histories.png",
    mark_until: int = 120,
) -> None:
    """Per-epoch training accuracy for each dataset, with validation as a dashed line."""
    present = _present(data)
    fig, axes = _panels(len(present), (10.4, 3.8), (10.4, 6.8))
    for ax, dataset in zip(axes, present):
        for row in _rows(data, dataset):
            epochs = list(range(1, len(row["train"]) + 1))
            color = SEED_COLORS[int(row["seed"])]
            ax.plot(epochs, [100.0 * float(value) for value in row["train"]], color=color, linewidth=1.3, label=f"Seed {row['seed']} train")
            ax.plot(
                epochs,
                [100.0 * float(value) for value in row["val"]],
                color=color,
                linewidth=1.0,
                linestyle="--",
                label=f"Seed {row['seed']} validation",
            )
        for boundary in range(10, mark_until, 10):
            ax.axvline(boundary, color="#e2e8f0", linewidth=1, zorder=0)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_test_histories(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-test-by-generation.png",
) -> None:
    """Official test after each generation, one panel per dataset."""
    present = _present(data)
    fig, axes = _panels(len(present), (10.4, 3.8), (10.4, 6.6))
    for ax, dataset in zip(axes, present):
        for row in _rows(data, dataset):
            ax.plot(
                row["test_epoch"],
                [100.0 * float(value) for value in row["test"]],
                color=SEED_COLORS[int(row["seed"])],
                marker="o",
                linewidth=1.4,
                label=f"Seed {row['seed']}",
            )
        reference = REFERENCE_TEST[dataset]
        if reference is not None:
            ax.axhline(reference, color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Official test accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_parameters(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-parameters.png",
    scale: int = 1_000_000,
    ylabel: str = "Parameters (millions)",
) -> None:
    """Parameter count at each generation end."""
    present = _present(data)
    fig, axes = _panels(len(present), (10.4, 3.8), (10.4, 6.6))
    for ax, dataset in zip(axes, present):
        for row in _rows(data, dataset):
            ax.plot(
                row["test_epoch"],
                [int(value) / scale for value in row["params_generation"]],
                color=SEED_COLORS[int(row["seed"])],
                marker="o",
                linewidth=1.4,
                label=f"Seed {row['seed']}",
            )
        ax.set_title(DATASET_LABEL[dataset])
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def write_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write the Experiment 009 figures and refresh the snapshot from history.pt when those files exist."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    run_dir = run_dir or RUN_DIR
    finished = all((run_dir / dataset / "seed_1" / "history.pt").is_file() for dataset in DATASETS)
    if finished:
        data = load_measurement(run_dir)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_train_histories(data, out_dir)
    plot_final_test(data, out_dir)
    plot_test_histories(data, out_dir)
    plot_parameters(data, out_dir)


def plot_cifar10_seed(data: dict[str, object], out_dir: Path, seed: int = 2) -> None:
    """Train, validation, and official test for one CIFAR-10 seed, using every saved epoch."""
    match = next((row for row in _rows(data, "cifar10") if int(row["seed"]) == seed), None)
    if match is None:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=140)
    epochs = list(range(1, len(match["train"]) + 1))
    ax.plot(epochs, [100.0 * float(value) for value in match["train"]], color="#d97706", linewidth=1.4, label="Train")
    ax.plot(epochs, [100.0 * float(value) for value in match["val"]], color="#d97706", linewidth=1.1, linestyle="--", label="Validation")
    ax.plot(match["test_epoch"], [100.0 * float(value) for value in match["test"]], color="#1d4ed8", marker="o", linewidth=1.4, label="Official test")
    ax.axhline(91.21, color="#64748b", linestyle="--", linewidth=1, label="91.21% reference")
    for boundary in range(10, len(epochs), 10):
        ax.axvline(boundary, color="#e2e8f0", linewidth=1, zorder=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color="#e2e8f0")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "009-small20-cifar10-seed2.png")
    plt.close(fig)


def write_small_starter_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write the 20-generation one-convolution figures and refresh their snapshot."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or SMALL_SNAPSHOT
    run_dir = run_dir or SMALL_RUN_DIR
    finished = any((run_dir / dataset).glob("seed_*/history.pt") for dataset in DATASETS)
    if finished:
        data = load_measurement(run_dir)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_train_histories(data, out_dir, filename="009-small20-train-histories.png", mark_until=200)
    plot_final_test(data, out_dir, filename="009-small20-final-official-test.png")
    plot_test_histories(data, out_dir, filename="009-small20-test-by-generation.png")
    plot_parameters(
        data,
        out_dir,
        filename="009-small20-parameters.png",
        scale=1_000,
        ylabel="Parameters (thousands)",
    )
    plot_cifar10_seed(data, out_dir, seed=2)


if __name__ == "__main__":
    write_charts()
    write_small_starter_charts()
