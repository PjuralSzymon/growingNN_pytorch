"""Generate Experiment 007 charts from the finished run, or from the JSON snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SITE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-007-cifar10-test90-package.json"
RUN_DIR = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp007_test90_package"
)
SEEDS = (110, 111, 112)
GROWTH_EPOCHS = 7 * 8
FINISH_EPOCHS = 24 * 5
SEED_COLORS = {110: "#1d4ed8", 111: "#d97706", 112: "#059669"}
ORIGINAL_COLOR = "#6b7280"
ADDED_COLOR = "#c2410c"


def experiment_has_finished(run_dir: Path) -> bool:
    """True when every seed has an original history and an added finish checkpoint."""
    original = [run_dir / "original" / "original" / f"seed_{seed}" / "history.pt" for seed in SEEDS]
    added = [run_dir / "added" / f"seed_{seed}" / "added_finish.pt" for seed in SEEDS]
    return all(path.is_file() for path in original + added)


def _load_original(path: Path, seed: int) -> dict[str, object]:
    history = torch.load(path, map_location="cpu", weights_only=False)
    test = [float(value) for value in history["test_acc"]]
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
    params = [int(value) for value in history["param_count"]]
    return {
        "seed": seed,
        "train": [float(value) for value in history["train_acc"]],
        "val": [float(value) for value in history["val_acc"]],
        "test": test,
        "test_epoch": [10 * (index + 1) for index in range(len(test))],
        "final_test": test[-1],
        "final_train": float(history["train_acc"][-1]),
        "final_val": float(history["val_acc"][-1]),
        "params_start": params[0],
        "params_end": params[-1],
        "actions": actions,
    }


def _load_added(seed_dir: Path, seed: int) -> dict[str, object]:
    growth = torch.load(seed_dir / "added_growth.pt", map_location="cpu", weights_only=False)
    best = torch.load(seed_dir / "added_finish_best.pt", map_location="cpu", weights_only=False)
    final = torch.load(seed_dir / "added_finish.pt", map_location="cpu", weights_only=False)
    test = [float(value) for value in growth["test_acc"]]
    params_end = int(sum(tensor.numel() for tensor in growth["state_dict"].values()))
    seq_conv_count = len(
        {
            key.split(".")[0]
            for key in growth["state_dict"]
            if key.startswith("seq_conv_") and key.endswith(".weight") and "_bn." not in key
        }
    )
    best_chunk = int(best["chunk"])
    return {
        "seed": seed,
        "growth_test": test,
        "growth_test_epoch": [8 * (index + 1) for index in range(len(test))],
        "growth_final_test": test[-1],
        "best_test": float(best["test_acc"]),
        "best_chunk": best_chunk,
        "best_epoch": GROWTH_EPOCHS + 5 * (best_chunk + 1),
        "final_test": float(final["test_acc"]),
        "final_epoch": GROWTH_EPOCHS + FINISH_EPOCHS,
        "params_end": params_end,
        "seq_conv_count": seq_conv_count,
    }


def load_from_run(run_dir: Path) -> dict[str, object]:
    """Read the three original histories and the three added checkpoints."""
    if not experiment_has_finished(run_dir):
        raise FileNotFoundError(
            "Experiment 007 has not finished. Run "
            "experiments/train_cifar10_exp007_test90_package.py before generating charts."
        )
    return {
        "experiment": "007",
        "seeds": list(SEEDS),
        "original": [
            _load_original(run_dir / "original" / "original" / f"seed_{seed}" / "history.pt", seed)
            for seed in SEEDS
        ],
        "added": [
            _load_added(run_dir / "added" / f"seed_{seed}", seed)
            for seed in SEEDS
        ],
    }


def _load_added_epoch_history(path: Path, seed: int) -> dict[str, object]:
    """Read one added history.pt. Train and validation are per epoch. Test uses the saved epochs."""
    history = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "seed": seed,
        "train": [float(value) for value in history["train_acc"]],
        "val": [float(value) for value in history["val_acc"]],
        "test": [float(value) for value in history["test_acc"]],
        "test_epoch": [int(value) for value in history["test_epoch"]],
    }


def load_added_epoch_histories(run_dir: Path) -> list[dict[str, object]]:
    """Return added seeds whose history.pt is already on disk."""
    rows = []
    for seed in SEEDS:
        path = run_dir / "added" / f"seed_{seed}" / "history.pt"
        if path.is_file():
            rows.append(_load_added_epoch_history(path, seed))
    return rows


def _added_history_count(run_dir: Path) -> int:
    return sum((run_dir / "added" / f"seed_{seed}" / "history.pt").is_file() for seed in SEEDS)


def load_measurement(run_dir: Path | None = None, snapshot_path: Path | None = None) -> dict[str, object]:
    """Prefer the finished run. Fall back to the committed snapshot."""
    run_dir = run_dir or RUN_DIR
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    if experiment_has_finished(run_dir):
        return load_from_run(run_dir)
    if snapshot_path.is_file():
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "Experiment 007 has not finished and no snapshot exists. Run "
        "experiments/train_cifar10_exp007_test90_package.py before generating charts."
    )


def plot_final_bars(data: dict[str, object], out_dir: Path) -> None:
    """Final official test of the original group, and best and last added finish points."""
    original = list(data["original"])
    added = list(data["added"])
    labels = [str(row["seed"]) for row in original]
    original_values = [100.0 * float(row["final_test"]) for row in original]
    best_values = [100.0 * float(row["best_test"]) for row in added]
    final_values = [100.0 * float(row["final_test"]) for row in added]
    positions = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    ax.bar([item - width for item in positions], original_values, width, color=ORIGINAL_COLOR, label="Original, last generation")
    ax.bar(positions, best_values, width, color=ADDED_COLOR, label="Added, best finish chunk")
    ax.bar([item + width for item in positions], final_values, width, color="#f5c58a", label="Added, last finish chunk")
    ax.axhline(90, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Seed {label}" for label in labels])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_title("Official test accuracy by seed")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "007-final-official-test.png")
    plt.close(fig)


def _plot_seed_series(ax, rows: list[dict[str, object]], epoch_key: str, value_key: str, epoch_from_length: bool) -> None:
    for row in rows:
        values = [100.0 * float(value) for value in row[value_key]]
        if epoch_from_length:
            epochs = list(range(1, len(values) + 1))
            marker = None
        else:
            epochs = [int(value) for value in row[epoch_key]]
            marker = "o"
        ax.plot(
            epochs,
            values,
            color=SEED_COLORS[int(row["seed"])],
            marker=marker,
            linewidth=1.4,
            label=f"Seed {row['seed']}",
        )


def _mark_unsaved(ax, message: str) -> None:
    ax.set_xticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9)


def plot_history_grid(
    data: dict[str, object],
    out_dir: Path,
    added_epochs: list[dict[str, object]] | None = None,
) -> None:
    """Train, official test, and validation for both groups. Only series present in the run files are drawn."""
    original = list(data["original"])
    added = list(data["added"])
    epoch_rows = added_epochs if added_epochs is not None else [row for row in added if "train" in row]
    fig, axes = plt.subplots(3, 2, figsize=(11.6, 8.6), dpi=140, sharey="row")
    _plot_seed_series(axes[0, 0], original, "", "train", True)
    _plot_seed_series(axes[1, 0], original, "test_epoch", "test", False)
    _plot_seed_series(axes[2, 0], original, "", "val", True)
    if epoch_rows:
        _plot_seed_series(axes[0, 1], epoch_rows, "", "train", True)
        _plot_seed_series(axes[1, 1], epoch_rows, "test_epoch", "test", False)
        _plot_seed_series(axes[2, 1], epoch_rows, "", "val", True)
        axes[0, 1].set_title("Added, seed " + ", ".join(str(row["seed"]) for row in epoch_rows))
        axes[0, 1].legend(frameon=False, fontsize=8)
        axes[0, 1].grid(True, axis="y", color="#e2e8f0")
        axes[2, 1].grid(True, axis="y", color="#e2e8f0")
        axes[2, 1].set_xlabel("Epoch")
    else:
        _mark_unsaved(axes[0, 1], "Not in the run files.\nPer-epoch train accuracy was not saved.")
        _plot_seed_series(axes[1, 1], added, "growth_test_epoch", "growth_test", False)
        _mark_unsaved(axes[2, 1], "Not in the run files.\nPer-epoch validation accuracy was not saved.")
    axes[0, 0].set_title("Original")
    if not epoch_rows:
        axes[0, 1].set_title("Added")
    axes[0, 0].set_ylabel("Train accuracy (%)")
    axes[1, 0].set_ylabel("Official test accuracy (%)")
    axes[2, 0].set_ylabel("Validation accuracy (%)")
    axes[2, 0].set_xlabel("Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 0].set_ylim(40, 100)
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1], axes[2, 0]):
        ax.grid(True, axis="y", color="#e2e8f0")
    axes[1, 0].axhline(90, color="#64748b", linestyle="--", linewidth=1)
    axes[1, 1].axhline(90, color="#64748b", linestyle="--", linewidth=1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "007-history-grid.png")
    plt.close(fig)


def write_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write the Experiment 007 figures. Refresh the snapshot when the run is present."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    run_dir = run_dir or RUN_DIR
    partial_history = 0 < _added_history_count(run_dir) < len(SEEDS)
    if partial_history and snapshot_path.is_file():
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        data = load_measurement(run_dir, snapshot_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if experiment_has_finished(run_dir) and not partial_history:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_final_bars(data, out_dir)
    plot_history_grid(data, out_dir, load_added_epoch_histories(run_dir))


if __name__ == "__main__":
    write_charts()
