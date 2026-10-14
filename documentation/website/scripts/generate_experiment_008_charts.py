"""Generate Experiment 008 charts from the finished run, or from the JSON snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SITE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-008-cifar10-generation-only.json"
RUN_DIR = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp008_generation_only"
)
SEEDS = (110, 111, 112)
SEED_COLORS = {110: "#1d4ed8", 111: "#d97706", 112: "#059669"}


def experiment_has_finished(run_dir: Path) -> bool:
    """True when every seed has a generation-loop history.pt."""
    return all((run_dir / f"seed_{seed}" / "history.pt").is_file() for seed in SEEDS)


def _load_history(path: Path, seed: int) -> dict[str, object]:
    """Read one seed. Train and validation are per epoch. Official test uses the saved epochs."""
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
    best_generation = max(range(len(test)), key=test.__getitem__)
    return {
        "seed": seed,
        "train": [float(value) for value in history["train_acc"]],
        "val": [float(value) for value in history["val_acc"]],
        "lr": [float(value) for value in history["lr"]],
        "param_count": params,
        "test": test,
        "test_epoch": [int(value) for value in history["test_epoch"]],
        "final_train": float(history["train_acc"][-1]),
        "final_val": float(history["val_acc"][-1]),
        "final_test": test[-1],
        "best_test": test[best_generation],
        "best_generation": best_generation,
        "params_start": params[0],
        "params_end": params[-1],
        "actions": actions,
    }


def load_from_run(run_dir: Path) -> dict[str, object]:
    """Read the three generation-loop histories."""
    if not experiment_has_finished(run_dir):
        raise FileNotFoundError(
            "Experiment 008 has not finished. Run "
            "experiments/train_cifar10_exp008_generation_only.py before generating charts."
        )
    return {
        "experiment": "008",
        "seeds": list(SEEDS),
        "runs": [_load_history(run_dir / f"seed_{seed}" / "history.pt", seed) for seed in SEEDS],
    }


def load_measurement(run_dir: Path | None = None, snapshot_path: Path | None = None) -> dict[str, object]:
    """Prefer the finished run. Fall back to the committed snapshot."""
    run_dir = run_dir or RUN_DIR
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    if experiment_has_finished(run_dir):
        return load_from_run(run_dir)
    if snapshot_path.is_file():
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "Experiment 008 has not finished and no snapshot exists. Run "
        "experiments/train_cifar10_exp008_generation_only.py before generating charts."
    )


def _plot_seed_series(ax, rows: list[dict[str, object]], value_key: str, epoch_key: str | None) -> None:
    for row in rows:
        values = [100.0 * float(value) for value in row[value_key]]
        if epoch_key is None:
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


def plot_official_test(data: dict[str, object], out_dir: Path) -> None:
    """Official test accuracy at each generation end."""
    rows = list(data["runs"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    _plot_seed_series(ax, rows, "test", "test_epoch")
    ax.axhline(90, color="#64748b", linestyle="--", linewidth=1)
    ax.set_ylim(50, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_title("Official test accuracy by generation")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-official-test.png")
    plt.close(fig)


def plot_history(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for the three seeds."""
    rows = list(data["runs"])
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), dpi=140, sharex=True)
    _plot_seed_series(axes[0], rows, "train", None)
    _plot_seed_series(axes[1], rows, "test", "test_epoch")
    _plot_seed_series(axes[2], rows, "val", None)
    axes[0].set_ylabel("Train accuracy (%)")
    axes[1].set_ylabel("Official test accuracy (%)")
    axes[2].set_ylabel("Validation accuracy (%)")
    axes[2].set_xlabel("Epoch")
    axes[1].axhline(90, color="#64748b", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_ylim(30, 100)
        ax.grid(True, axis="y", color="#e2e8f0")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "008-history.png")
    plt.close(fig)


def plot_learning_rate(data: dict[str, object], out_dir: Path) -> None:
    """Saved learning rate. One line when every seed stored the same schedule."""
    rows = list(data["runs"])
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=140)
    same = all(row["lr"] == rows[0]["lr"] for row in rows[1:])
    drawn = rows[:1] if same else rows
    for row in drawn:
        epochs = list(range(1, len(row["lr"]) + 1))
        label = "All seeds" if same else f"Seed {row['seed']}"
        color = "#1d4ed8" if same else SEED_COLORS[int(row["seed"])]
        ax.plot(epochs, row["lr"], color=color, linewidth=1.4, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-learning-rate.png")
    plt.close(fig)


def write_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write the Experiment 008 figures. Refresh the snapshot when the run is present."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    run_dir = run_dir or RUN_DIR
    data = load_measurement(run_dir, snapshot_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if experiment_has_finished(run_dir):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_official_test(data, out_dir)
    plot_history(data, out_dir)
    plot_learning_rate(data, out_dir)


if __name__ == "__main__":
    write_charts()
