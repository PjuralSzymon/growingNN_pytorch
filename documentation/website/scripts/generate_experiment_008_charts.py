"""Generate Experiment 008 charts from the finished run, or from the JSON snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
PHASE10_SEEDS = (110, 111, 112, 113, 114)
PHASE10_ARMS = ("base", "lr", "conv", "lr_conv")
SEED_COLORS = {
    110: "#1d4ed8",
    111: "#d97706",
    112: "#059669",
    113: "#7c3aed",
    114: "#e11d48",
}
ARM_COLORS = {
    "base": "#64748b",
    "lr": "#1d4ed8",
    "conv": "#d97706",
    "lr_conv": "#059669",
}
# Seed 112 is the highest last official test. These edges are the Generation action
# lines in growingnn.log from 2026-10-14 09:15 to 09:41, matched to that seed's
# saved validation accuracy before each action.
SEED_112_INSERTS = (
    (0, "conv2", "linear", 0.6266),
    (1, "seq_conv_0", "linear", 0.7256),
    (2, "seq_conv_1", "linear", 0.8088),
    (3, "seq_conv_2", "linear", 0.8356),
    (4, "seq_conv_3", "linear", 0.8490),
    (5, "seq_conv_3", "seq_conv_4", 0.8692),
    (6, "seq_conv_4", "linear", 0.8754),
    (7, "seq_conv_3", "seq_conv_5", 0.8852),
    (8, "seq_conv_6", "linear", 0.8926),
    (9, "seq_conv_0", "seq_conv_1", 0.9046),
    (10, "seq_conv_0", "seq_conv_9", 0.9068),
)
ARCHITECTURE_STAGES = (
    (0, "008-architecture-seed112-start.png", "Start, generation 0"),
    (4, "008-architecture-seed112-gen3.png", "After generation 3"),
    (11, "008-architecture-seed112-final.png", "After generation 10"),
)


def experiment_has_finished(run_dir: Path) -> bool:
    """True when every seed has a generation-loop history.pt."""
    return all((run_dir / f"seed_{seed}" / "history.pt").is_file() for seed in SEEDS)


def _load_history(path: Path, seed: int) -> dict[str, object]:
    """Read one seed. Train and validation are per epoch. Official test uses the saved epochs."""
    history = torch.load(path, map_location="cpu", weights_only=False)
    test = [float(value) for value in history["test_acc"]]
    froms = list(history.get("action_from") or [])
    tos = list(history.get("action_to") or [])
    actions = []
    for index, name in enumerate(history["action_name"]):
        actions.append(
            {
                "generation": int(history["action_generation"][index]),
                "name": str(name),
                "from": str(froms[index]) if index < len(froms) else "",
                "to": str(tos[index]) if index < len(tos) else "",
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
    data = {
        "experiment": "008",
        "seeds": list(SEEDS),
        "runs": [_load_history(run_dir / f"seed_{seed}" / "history.pt", seed) for seed in SEEDS],
    }
    _attach_seed_112_inserts(data)
    return data


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


def _plot_accuracy_grid(
    modes: tuple[tuple[str, object], ...],
    out_path: Path,
    *,
    ylim: tuple[float, float] = (0, 100),
) -> None:
    """One row per mode and one column each for train, official test, and validation."""
    if any(not rows for _, rows in modes):
        return
    metrics = (
        ("train", None, "Train accuracy"),
        ("test", "test_epoch", "Official test accuracy"),
        ("val", None, "Validation accuracy"),
    )
    fig, axes = plt.subplots(
        len(modes),
        3,
        figsize=(11.4, 2.55 * len(modes) + 0.8),
        dpi=140,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, (mode, rows) in enumerate(modes):
        for col_index, (key, epoch_key, title) in enumerate(metrics):
            ax = axes[row_index][col_index]
            _plot_seed_series(ax, list(rows), key, epoch_key)
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(f"{mode}\nAccuracy (%)")
            if key == "test":
                ax.axhline(90, color="#64748b", linestyle="--", linewidth=1)
            ax.set_ylim(*ylim)
            ax.grid(True, axis="y", color="#e2e8f0")
        if row_index == 0 and len(list(rows)) <= 3:
            axes[0][0].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    if len(list(modes[0][1])) > 3:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_history(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for the phase 1 seeds, in one row."""
    _plot_accuracy_grid((("Only convolution", data["runs"]),), out_dir / "008-history.png")


def _attach_seed_112_inserts(data: dict[str, object]) -> None:
    """Attach the logged seed-112 insert list after it matches that seed's saved validation."""
    row = next(item for item in data["runs"] if int(item["seed"]) == 112)
    befores = [float(action["val_before"]) for action in row["actions"]]
    expected = [item[3] for item in SEED_112_INSERTS]
    if len(befores) != len(expected):
        return
    if any(abs(got - want) > 1e-6 for got, want in zip(befores, expected)):
        raise ValueError("Seed 112 action validation does not match the logged insert list.")
    data["best_seed"] = 112
    data["best_seed_inserts"] = [
        {"generation": generation, "from": src, "to": dst}
        for generation, src, dst, _val in SEED_112_INSERTS
    ]


def plot_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count by epoch. One line when every seed stored the same list."""
    rows = list(data["runs"])
    if any("param_count" not in row for row in rows):
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    same = all(row["param_count"] == rows[0]["param_count"] for row in rows[1:])
    drawn = rows[:1] if same else rows
    for row in drawn:
        epochs = list(range(1, len(row["param_count"]) + 1))
        label = "Seeds 110, 111, and 112" if same else f"Seed {row['seed']}"
        color = "#1d4ed8" if same else SEED_COLORS[int(row["seed"])]
        ax.plot(epochs, [int(value) for value in row["param_count"]], color=color, linewidth=1.6, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter count")
    ax.set_title("Parameter count")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-parameters.png")
    plt.close(fig)


def write_architecture(data: dict[str, object], out_dir: Path) -> None:
    """Draw the seed-112 FX graph at the start, after generation 3, and at the end."""
    inserts = data.get("best_seed_inserts")
    if not inserts:
        return
    from experiments.train_cifar10_exp007_test90_package import FourLayerCifarNet
    from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
    from growingnn.core.traced_model import TracedModel
    from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph

    stage_names = {count: filename for count, filename, _title in ARCHITECTURE_STAGES}
    traced = TracedModel.create(FourLayerCifarNet(channels=128, hidden_dim=256), (1, 3, 32, 32))

    def _draw(count: int) -> None:
        filename = stage_names.get(count)
        if filename is not None:
            draw_filtered_fx_graph(traced.gm, str(out_dir / filename.removesuffix(".png")), fmt="png")

    _draw(0)
    for index, insert in enumerate(list(inserts), start=1):
        actions = AddSeqConvLayer.generate_all_actions(traced, add_batchnorm_and_relu=True, before_global_pool=True)
        match = [
            action for action in actions
            if action.params[0] == insert["from"] and action.params[1] == insert["to"]
        ]
        if len(match) != 1:
            raise ValueError(f"Insert {insert['from']} -> {insert['to']} matched {len(match)} actions.")
        match[0].execute(traced)
        traced.invalidate()
        _draw(index)


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


def _saved_group(group_dir: Path, seeds: tuple[int, ...] = SEEDS) -> list[dict[str, object]]:
    """Load the seeds of one group that already have history.pt."""
    rows = []
    for seed in seeds:
        path = group_dir / f"seed_{seed}" / "history.pt"
        if path.is_file():
            rows.append(_load_history(path, seed))
    return rows


def _last_percent(row: dict[str, object], key: str) -> float:
    """Return one seed's final accuracy for train, official test, or validation, in percent."""
    if key == "test":
        return 100.0 * float(row["final_test"] if "final_test" in row else row["test"][-1])
    if key == "val":
        return 100.0 * float(row["final_val"] if "final_val" in row else row["val"][-1])
    return 100.0 * float(row["final_train"] if "final_train" in row else row["train"][-1])


def plot_phase2_group_means(data: dict[str, object], out_dir: Path) -> None:
    """Mean final train, official test, and validation for phase 1 and the three phase 2 groups."""
    groups = (
        ("Only convolution", data.get("runs")),
        ("All", data.get("phase2_all")),
        ("Grow", data.get("phase2_grow")),
        ("Seq and residual", data.get("phase2_seq_res")),
    )
    if any(not rows for _, rows in groups):
        return
    series = (
        ("Mean training", "train", "#1d4ed8"),
        ("Mean official test", "test", "#059669"),
        ("Mean validation", "val", "#d97706"),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    centers = list(range(len(groups)))
    width = 0.24
    seed_labeled = False
    for index, (label, key, color) in enumerate(series):
        offset = (index - 1) * width
        heights = []
        positions = []
        for center, (_, rows) in zip(centers, groups):
            values = [_last_percent(row, key) for row in rows]
            position = center + offset
            heights.append(sum(values) / len(values))
            positions.append(position)
            ax.scatter(
                [position] * len(values),
                values,
                s=22,
                color="#0f172a",
                zorder=3,
                label="Seed" if not seed_labeled else None,
            )
            seed_labeled = True
        ax.bar(positions, heights, width=width, color=color, label=label, zorder=2)
    ax.set_xticks(centers)
    ax.set_xticklabels([name for name, _rows in groups])
    ax.set_ylim(75, 100)
    ax.set_ylabel("Accuracy at epoch 120 (%)")
    ax.set_title("Mean final accuracy by action group")
    ax.axhline(90, color="#64748b", linestyle="--", linewidth=1)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase2-group-means.png")
    plt.close(fig)


def plot_phase2_mode_grid(data: dict[str, object], out_dir: Path) -> None:
    """Four action groups by three accuracy curves. Each seed is its own line."""
    _plot_accuracy_grid(
        (
            ("Only convolution", data.get("runs")),
            ("All", data.get("phase2_all")),
            ("Grow", data.get("phase2_grow")),
            ("Seq and residual", data.get("phase2_seq_res")),
        ),
        out_dir / "008-phase2-mode-grid.png",
    )


def plot_phase2_param_grid(data: dict[str, object], out_dir: Path) -> None:
    """One parameter-count panel per action group. Each seed is its own line."""
    modes = (
        ("Only convolution", data.get("runs")),
        ("All", data.get("phase2_all")),
        ("Grow", data.get("phase2_grow")),
        ("Seq and residual", data.get("phase2_seq_res")),
    )
    if any(not rows for _, rows in modes):
        return
    if any("param_count" not in row for _, rows in modes for row in rows):
        return
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), dpi=140, sharex=True, sharey=True)
    for ax, (mode, rows) in zip(axes, modes):
        for row in rows:
            epochs = list(range(1, len(row["param_count"]) + 1))
            ax.plot(
                epochs,
                [int(value) for value in row["param_count"]],
                color=SEED_COLORS[int(row["seed"])],
                linewidth=1.6,
                label=f"Seed {row['seed']}",
            )
        ax.set_title(mode)
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value / 1_000_000:.1f}"))
    axes[0].set_ylabel("Parameter count (millions)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase2-param-grid.png")
    plt.close(fig)


def plot_phase2_official_test(data: dict[str, object], out_dir: Path) -> None:
    """Official test for phase 1 and for the phase 2 all group, one color per seed."""
    phase2 = data.get("phase2_all")
    if not phase2:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    for row in data["runs"]:
        epochs = [int(value) for value in row["test_epoch"]]
        values = [100.0 * float(value) for value in row["test"]]
        ax.plot(
            epochs,
            values,
            color=SEED_COLORS[int(row["seed"])],
            linestyle="--",
            linewidth=1.2,
            label=f"Only convolution {row['seed']}",
        )
    for row in phase2:
        epochs = [int(value) for value in row["test_epoch"]]
        values = [100.0 * float(value) for value in row["test"]]
        ax.plot(
            epochs,
            values,
            color=SEED_COLORS[int(row["seed"])],
            marker="o",
            linewidth=1.6,
            label=f"All seed {row['seed']}",
        )
    ax.axhline(90, color="#64748b", linestyle=":", linewidth=1)
    ax.set_ylim(50, 100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_title("Official test accuracy, only convolution and all")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase2-official-test.png")
    plt.close(fig)


def plot_phase2_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for the phase 2 all group. Each seed has its own list."""
    phase2 = data.get("phase2_all")
    if not phase2 or any("param_count" not in row for row in phase2):
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    for row in phase2:
        epochs = list(range(1, len(row["param_count"]) + 1))
        ax.plot(
            epochs,
            [int(value) for value in row["param_count"]],
            color=SEED_COLORS[int(row["seed"])],
            linewidth=1.6,
            label=f"Seed {row['seed']}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter count")
    ax.set_title("Parameter count, phase 2 all")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase2-parameters.png")
    plt.close(fig)


def plot_phase3_mode_grid(data: dict[str, object], out_dir: Path) -> None:
    """Only convolution, short all, and long all, each with train, official test, and validation."""
    _plot_accuracy_grid(
        (
            ("Only convolution", data.get("runs")),
            ("All, short", data.get("phase2_all")),
            ("All, long", data.get("phase3_all")),
        ),
        out_dir / "008-phase3-mode-grid.png",
    )


def plot_phase4_mode_grid(data: dict[str, object], out_dir: Path) -> None:
    """Only convolution and the finished 1-epoch phase 4 grade, with three accuracy columns."""
    rows = data.get("phase4_epochs_1")
    if not rows or not data.get("runs"):
        return
    _plot_accuracy_grid(
        (
            ("Only convolution", data.get("runs")),
            ("1 epoch", rows),
        ),
        out_dir / "008-phase4-mode-grid.png",
    )


def plot_phase5_mode_grid(data: dict[str, object], out_dir: Path) -> None:
    """Only convolution, the full augmented grade, and the 5,000-image clean grade."""
    rows = data.get("phase5_epochs_1")
    augmented = data.get("phase4_epochs_1")
    if not rows or not augmented or not data.get("runs"):
        return
    _plot_accuracy_grid(
        (
            ("Only convolution", data.get("runs")),
            ("45,000 augmented", augmented),
            ("5,000 clean", rows),
        ),
        out_dir / "008-phase5-mode-grid.png",
    )


_ACTION_LABELS = {
    "AddSeqConvLayer": "Seq conv",
    "AddResConvLayer": "Res conv",
    "AddSeqLinearLayer": "Seq linear",
    "AddResLinearLayer": "Res linear",
    "AddNeurons": "Add neurons",
    "DelNeurons": "Delete neurons",
    "AddSeqDropoutLayer": "Dropout",
    "DelLayer": "Delete layer",
}


def _labeled_grades(
    data: dict[str, object],
    keys: tuple[tuple[str, str], ...],
    required: str,
) -> list[tuple[str, list[dict[str, object]]]]:
    """Saved grades that contain one field, in the given order."""
    grades = []
    for key, label in keys:
        rows = data.get(key)
        if rows and all(required in row for row in rows):
            grades.append((label, list(rows)))
    return grades


def _phase5_grades(data: dict[str, object], required: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Saved phase 5 grades that contain one field, in simulation-epoch order."""
    keys = tuple(
        (f"phase5_epochs_{epochs}", "1 epoch" if epochs == 1 else f"{epochs} epochs")
        for epochs in (1, 2, 4, 5, 6, 7)
    )
    return _labeled_grades(data, keys, required)


def _phase6_grades(data: dict[str, object], required: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Saved phase 6 grades that contain one field, from the largest clean prefix down."""
    keys = tuple(
        (f"phase6_size_{size}", f"{size:,} images")
        for size in (5000, 2500, 1250, 625, 312, 156, 78)
    )
    return _labeled_grades(data, keys, required)


def plot_phase5_epoch_grid(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for each saved phase 5 simulation length."""
    grades = _phase5_grades(data, "test")
    if not grades:
        return
    _plot_accuracy_grid(tuple(grades), out_dir / "008-phase5-epoch-grid.png")


def plot_phase6_final_test(data: dict[str, object], out_dir: Path) -> None:
    """Last official test by clean prefix size. One line per seed, plus the three-seed mean."""
    grades = _phase6_grades(data, "final_test")
    if len(grades) < 2:
        return
    positions = list(range(len(grades)))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    seeds = [int(row["seed"]) for row in grades[0][1]]
    for seed in seeds:
        values = []
        for _label, rows in grades:
            row = next(item for item in rows if int(item["seed"]) == seed)
            values.append(100.0 * float(row["final_test"]))
        ax.plot(
            positions,
            values,
            color=SEED_COLORS[seed],
            marker="o",
            linewidth=1.6,
            alpha=0.35,
            label=f"Seed {seed}",
        )
    means = [
        sum(100.0 * float(row["final_test"]) for row in rows) / len(rows)
        for _label, rows in grades
    ]
    ax.plot(
        positions,
        means,
        color="#111827",
        marker="o",
        linewidth=2.2,
        zorder=3,
        label="Mean",
    )
    ax.axhline(90, color="#64748b", linestyle=":", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([label.replace(" images", "") for label, _rows in grades])
    ax.set_ylim(70, 95)
    ax.set_xlabel("Clean simulation images")
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_title("Official test at epoch 120")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Mean"), *[index for index, label in enumerate(labels) if label != "Mean"]]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        fontsize=8,
    )
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase6-final-test.png")
    plt.close(fig)


def plot_phase6_epoch_grid(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for each saved phase 6 clean-prefix size."""
    grades = _phase6_grades(data, "test")
    if not grades:
        return
    _plot_accuracy_grid(tuple(grades), out_dir / "008-phase6-epoch-grid.png")


def _phase7_grades(data: dict[str, object], required: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Phase 7 lengths, with the matching phase 6 10-epoch grade first."""
    keys = (("phase6_size_1250", "10 epochs"), ("phase7_epochs_20", "20 epochs"), ("phase7_epochs_30", "30 epochs"))
    return _labeled_grades(data, keys, required)


def plot_phase7_epoch_grid(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for the 1,250-image grade at each real training length."""
    grades = _phase7_grades(data, "test")
    if not grades:
        return
    _plot_accuracy_grid(tuple(grades), out_dir / "008-phase7-epoch-grid.png")


def _phase8_grades(data: dict[str, object], required: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Phase 8 generation counts, with the matching phase 6 12-generation grade included."""
    keys = (
        ("phase8_generations_10", "10 generations"),
        ("phase6_size_1250", "12 generations"),
        ("phase8_generations_15", "15 generations"),
        ("phase8_generations_20", "20 generations"),
    )
    return _labeled_grades(data, keys, required)


def plot_phase8_epoch_grid(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for each real generation count on the 1,250-image grade."""
    grades = _phase8_grades(data, "test")
    if not grades:
        return
    _plot_accuracy_grid(tuple(grades), out_dir / "008-phase8-epoch-grid.png")


def plot_phase8_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for each phase 8 generation count. Each seed has its own line."""
    grades = _phase8_grades(data, "param_count")
    if grades:
        _plot_parameter_grades(grades, out_dir / "008-phase8-parameters.png")


def _phase10_grades(data: dict[str, object], required: str) -> list[tuple[str, list[dict[str, object]]]]:
    """Phase 10 arms that contain one field, in switch order."""
    keys = (
        ("phase10_base", "base"),
        ("phase10_lr", "lr"),
        ("phase10_conv", "conv"),
        ("phase10_lr_conv", "lr_conv"),
    )
    return _labeled_grades(data, keys, required)


def plot_phase10_epoch_grid(data: dict[str, object], out_dir: Path) -> None:
    """Train, official test, and validation for each phase 10 arm."""
    grades = _phase10_grades(data, "test")
    if not grades:
        return
    _plot_accuracy_grid(tuple(grades), out_dir / "008-phase10-epoch-grid.png")


def plot_phase10_learning_rate(data: dict[str, object], out_dir: Path) -> None:
    """Learning rate for each phase 10 arm. One line when every seed stored the same schedule."""
    grades = _phase10_grades(data, "lr")
    if not grades:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), dpi=140, sharex=True, sharey=True, squeeze=False)
    flat = list(axes.ravel())
    for ax, (label, rows) in zip(flat, grades):
        same = all(row["lr"] == rows[0]["lr"] for row in rows[1:])
        drawn = rows[:1] if same else rows
        for row in drawn:
            epochs = list(range(1, len(row["lr"]) + 1))
            ax.plot(
                epochs,
                [float(value) for value in row["lr"]],
                color="#1d4ed8" if same else SEED_COLORS[int(row["seed"])],
                linewidth=1.6,
                label="All seeds" if same else f"Seed {row['seed']}",
            )
        ax.set_title(label)
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=7)
    for ax in flat[len(grades):]:
        ax.axis("off")
    flat[0].set_ylabel("Learning rate")
    flat[2].set_ylabel("Learning rate")
    flat[2].set_xlabel("Epoch")
    flat[3].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase10-learning-rate.png")
    plt.close(fig)


def plot_phase10_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for each phase 10 arm. Each seed has its own line."""
    grades = _phase10_grades(data, "param_count")
    if grades:
        _plot_parameter_grades(grades, out_dir / "008-phase10-parameters.png")


def plot_phase10_final_test(data: dict[str, object], out_dir: Path) -> None:
    """Last official test by arm. One line per seed, plus the five-seed mean."""
    grades = _phase10_grades(data, "final_test")
    if len(grades) < 2:
        return
    positions = list(range(len(grades)))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    seeds = [int(row["seed"]) for row in grades[0][1]]
    for seed in seeds:
        values = []
        for _label, rows in grades:
            row = next(item for item in rows if int(item["seed"]) == seed)
            values.append(100.0 * float(row["final_test"]))
        ax.plot(
            positions,
            values,
            color=SEED_COLORS[seed],
            marker="o",
            linewidth=1.6,
            alpha=0.35,
            label=f"Seed {seed}",
        )
    means = [
        sum(100.0 * float(row["final_test"]) for row in rows) / len(rows)
        for _label, rows in grades
    ]
    ax.plot(positions, means, color="#111827", marker="o", linewidth=2.2, zorder=3, label="Mean")
    ax.axhline(90, color="#64748b", linestyle=":", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _rows in grades])
    ax.set_ylim(75, 95)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_title("Official test at epoch 150")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Mean"), *[index for index, label in enumerate(labels) if label != "Mean"]]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        fontsize=8,
    )
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase10-final-test.png")
    plt.close(fig)


def plot_phase10_mean_curves(data: dict[str, object], out_dir: Path) -> None:
    """Five-seed mean of train, official test, and validation for each phase 10 arm."""
    grades = _phase10_grades(data, "test")
    if not grades:
        return
    metrics = (
        ("train", None, "Mean train accuracy"),
        ("test", "test_epoch", "Mean official test accuracy"),
        ("val", None, "Mean validation accuracy"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2), dpi=140, sharey=True, squeeze=False)
    for ax, (key, epoch_key, title) in zip(axes[0], metrics):
        for label, rows in grades:
            series = [list(row[key]) for row in rows]
            length = min(len(values) for values in series)
            mean = [sum(values[index] for values in series) / len(series) for index in range(length)]
            if epoch_key is None:
                epochs = list(range(1, length + 1))
                marker = None
            else:
                epochs = [int(value) for value in rows[0][epoch_key][:length]]
                marker = "o"
            ax.plot(
                epochs,
                [100.0 * value for value in mean],
                color=ARM_COLORS[label],
                marker=marker,
                linewidth=1.8,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color="#e2e8f0")
        if key == "test":
            ax.axhline(90, color="#64748b", linestyle="--", linewidth=1)
    axes[0][0].set_ylabel("Accuracy (%)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase10-mean-curves.png")
    plt.close(fig)


def _plot_action_counts(grades: list[tuple[str, list[dict[str, object]]]], out_path: Path) -> None:
    """One panel per grade. Each bar group is one executed action type."""
    tallies: list[list[dict[str, int]]] = []
    seen: list[str] = []
    for _label, rows in grades:
        grade_counts: list[dict[str, int]] = []
        for row in rows:
            tally: dict[str, int] = {}
            for action in row["actions"]:
                name = str(action["name"])
                tally[name] = tally.get(name, 0) + 1
                if name not in seen:
                    seen.append(name)
            grade_counts.append(tally)
        tallies.append(grade_counts)
    order = sorted(seen, key=lambda name: (-sum(tally.get(name, 0) for grade in tallies for tally in grade), name))
    if not order:
        return
    labels = [_ACTION_LABELS.get(name, name) for name in order]
    ncols = 3 if len(grades) > 3 else len(grades)
    nrows = (len(grades) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols + 0.4, 3.5 * nrows + 0.3),
        dpi=140,
        sharey=True,
        squeeze=False,
    )
    flat = list(axes.ravel())
    centers = list(range(len(order)))
    width = 0.24
    for ax, (label, rows), grade_counts in zip(flat, grades, tallies):
        for index, (row, tally) in enumerate(zip(rows, grade_counts)):
            offset = (index - 1) * width
            ax.bar(
                [center + offset for center in centers],
                [tally.get(name, 0) for name in order],
                width=width,
                color=SEED_COLORS[int(row["seed"])],
                label=f"Seed {row['seed']}",
            )
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(label)
        ax.set_ylim(0, 11)
        ax.grid(True, axis="y", color="#e2e8f0")
    for ax in flat[len(grades):]:
        ax.axis("off")
    flat[0].set_ylabel("Times chosen")
    flat[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_phase5_action_counts(data: dict[str, object], out_dir: Path) -> None:
    """How many times each seed chose each action type, one panel per saved grade."""
    grades = _phase5_grades(data, "actions")
    if grades:
        _plot_action_counts(grades, out_dir / "008-phase5-action-counts.png")


def plot_phase6_action_counts(data: dict[str, object], out_dir: Path) -> None:
    """How many times each seed chose each action type, one panel per saved prefix size."""
    grades = _phase6_grades(data, "actions")
    if grades:
        _plot_action_counts(grades, out_dir / "008-phase6-action-counts.png")


def _plot_parameter_grades(grades: list[tuple[str, list[dict[str, object]]]], out_path: Path) -> None:
    """Parameter count for each grade. Each seed has its own line."""
    ncols = 2 if len(grades) == 4 else (3 if len(grades) > 3 else len(grades))
    nrows = (len(grades) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols + 0.4, 3.2 * nrows + 0.3),
        dpi=140,
        sharey=True,
        squeeze=False,
    )
    flat = list(axes.ravel())
    for ax, (label, rows) in zip(flat, grades):
        for row in rows:
            epochs = list(range(1, len(row["param_count"]) + 1))
            ax.plot(
                epochs,
                [int(value) for value in row["param_count"]],
                color=SEED_COLORS[int(row["seed"])],
                linewidth=1.6,
                label=f"Seed {row['seed']}",
            )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="y", color="#e2e8f0")
    for ax in flat[len(grades):]:
        ax.axis("off")
    flat[0].set_ylabel("Parameter count")
    flat[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_phase5_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for each saved phase 5 grade. Each seed has its own line."""
    grades = _phase5_grades(data, "param_count")
    if grades:
        _plot_parameter_grades(grades, out_dir / "008-phase5-parameters.png")


def plot_phase6_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for each saved phase 6 grade. Each seed has its own line."""
    grades = _phase6_grades(data, "param_count")
    if grades:
        _plot_parameter_grades(grades, out_dir / "008-phase6-parameters.png")


def plot_phase4_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for the phase 4 one-epoch grade. Each seed has its own list."""
    rows = data.get("phase4_epochs_1")
    if not rows or any("param_count" not in row for row in rows):
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    for row in rows:
        epochs = list(range(1, len(row["param_count"]) + 1))
        ax.plot(
            epochs,
            [int(value) for value in row["param_count"]],
            color=SEED_COLORS[int(row["seed"])],
            linewidth=1.6,
            label=f"Seed {row['seed']}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter count")
    ax.set_title("Parameter count, 1 simulation epoch")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase4-parameters.png")
    plt.close(fig)


def plot_phase3_parameters(data: dict[str, object], out_dir: Path) -> None:
    """Parameter count for the phase 3 long all group. Each seed has its own list."""
    rows = data.get("phase3_all")
    if not rows or any("param_count" not in row for row in rows):
        return
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=140)
    for row in rows:
        epochs = list(range(1, len(row["param_count"]) + 1))
        ax.plot(
            epochs,
            [int(value) for value in row["param_count"]],
            color=SEED_COLORS[int(row["seed"])],
            linewidth=1.6,
            label=f"Seed {row['seed']}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter count")
    ax.set_title("Parameter count, long all")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(out_dir / "008-phase3-parameters.png")
    plt.close(fig)


def write_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
    phase2_dir: Path | None = None,
    phase3_dir: Path | None = None,
) -> None:
    """Write the Experiment 008 figures. Refresh the snapshot when the run is present."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    run_dir = run_dir or RUN_DIR
    phase2_dir = phase2_dir if phase2_dir is not None else run_dir.parent / "exp008_generation_only_phase2"
    phase3_dir = phase3_dir if phase3_dir is not None else run_dir.parent / "exp008_generation_only_phase3"
    data = load_measurement(run_dir, snapshot_path)
    for group in ("all", "grow", "seq_res"):
        rows = _saved_group(phase2_dir / group)
        if rows:
            data[f"phase2_{group}"] = rows
    phase3_rows = _saved_group(phase3_dir / "all")
    if phase3_rows:
        data["phase3_all"] = phase3_rows
    phase4_dir = run_dir.parent / "exp008_generation_only_phase4"
    phase5_dir = run_dir.parent / "exp008_generation_only_phase5"
    for sim_epochs in (1, 2, 4):
        rows = _saved_group(phase4_dir / f"epochs_{sim_epochs}")
        if len(rows) == len(SEEDS):
            data[f"phase4_epochs_{sim_epochs}"] = rows
    for sim_epochs in (1, 2, 4, 5, 6, 7):
        rows = _saved_group(phase5_dir / f"epochs_{sim_epochs}")
        if len(rows) == len(SEEDS):
            data[f"phase5_epochs_{sim_epochs}"] = rows
    phase6_dir = run_dir.parent / "exp008_generation_only_phase6"
    for size in (5000, 2500, 1250, 625, 312, 156, 78):
        rows = _saved_group(phase6_dir / f"size_{size}")
        if len(rows) == len(SEEDS):
            data[f"phase6_size_{size}"] = rows
    phase7_dir = run_dir.parent / "exp008_generation_only_phase7"
    for train_epochs in (20, 30):
        rows = _saved_group(phase7_dir / f"epochs_{train_epochs}")
        if len(rows) == len(SEEDS):
            data[f"phase7_epochs_{train_epochs}"] = rows
    phase8_dir = run_dir.parent / "exp008_generation_only_phase8"
    for n_generations in (10, 15, 20):
        rows = _saved_group(phase8_dir / f"generations_{n_generations}")
        if len(rows) == len(SEEDS):
            data[f"phase8_generations_{n_generations}"] = rows
    phase10_dir = run_dir.parent / "exp008_generation_only_phase10"
    for arm in PHASE10_ARMS:
        rows = _saved_group(phase10_dir / arm, PHASE10_SEEDS)
        if len(rows) == len(PHASE10_SEEDS):
            data[f"phase10_{arm}"] = rows
    out_dir.mkdir(parents=True, exist_ok=True)
    if experiment_has_finished(run_dir):
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_official_test(data, out_dir)
    plot_history(data, out_dir)
    plot_learning_rate(data, out_dir)
    plot_parameters(data, out_dir)
    write_architecture(data, out_dir)
    plot_phase2_official_test(data, out_dir)
    plot_phase2_parameters(data, out_dir)
    plot_phase2_group_means(data, out_dir)
    plot_phase2_mode_grid(data, out_dir)
    plot_phase2_param_grid(data, out_dir)
    plot_phase3_mode_grid(data, out_dir)
    plot_phase3_parameters(data, out_dir)
    plot_phase4_mode_grid(data, out_dir)
    plot_phase4_parameters(data, out_dir)
    plot_phase5_mode_grid(data, out_dir)
    plot_phase5_epoch_grid(data, out_dir)
    plot_phase5_action_counts(data, out_dir)
    plot_phase5_parameters(data, out_dir)
    plot_phase6_final_test(data, out_dir)
    plot_phase6_epoch_grid(data, out_dir)
    plot_phase6_action_counts(data, out_dir)
    plot_phase6_parameters(data, out_dir)
    plot_phase7_epoch_grid(data, out_dir)
    plot_phase8_epoch_grid(data, out_dir)
    plot_phase8_parameters(data, out_dir)
    plot_phase10_epoch_grid(data, out_dir)
    plot_phase10_learning_rate(data, out_dir)
    plot_phase10_parameters(data, out_dir)
    plot_phase10_final_test(data, out_dir)
    plot_phase10_mean_curves(data, out_dir)


if __name__ == "__main__":
    write_charts()
