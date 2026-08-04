"""Generate charts for Experiment 001: slope x logistic x model depth."""

from __future__ import annotations

import json
from collections import defaultdict
from math import atan, degrees
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


SITE = Path(__file__).parents[1]
DEFAULT_RUNS = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_mnist"
    / "runs"
    / "exp001_slope_logistic_model_depth"
)
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-001-slope-logistic-model-depth.json"

MODELS = ("big", "medium", "very_small")
MODEL_NAMES = {
    "big": "Big (420 params)",
    "medium": "Medium (276 params)",
    "very_small": "Very small (76 params)",
}
MODEL_COLORS = {
    "big": "#3568a8",
    "medium": "#4f8a63",
    "very_small": "#d18b2c",
}
ANGLES = ("2", "3", "4")
ANGLE_COLORS = {"2": "#3568a8", "3": "#4f8a63", "4": "#d18b2c"}
ANGLE_MARKERS = {"2": "o", "3": "s", "4": "^"}
SEED_COLORS = {100: "#3568a8", 101: "#4f8a63"}
ORDER_LABELS = ("1st", "2nd", "3rd", "4th", "5th+")


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics and metadata for every model-depth run."""
    runs: list[dict[str, object]] = []
    for main_path in sorted(runs_dir.rglob("board/main.json")):
        run_dir = main_path.parent.parent
        parts = run_dir.relative_to(runs_dir).parts
        metrics_path = main_path.parent / "metrics" / "training.json"
        main = json.loads(main_path.read_text(encoding="utf-8"))
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        actions = [
            (item["generation"], item["actionExecuted"])
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        runs.append(
            {
                "angle": parts[0].removeprefix("slope_").removesuffix("deg"),
                "model": parts[1],
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "elapsed_sec": main["trainingTimeElapsedSec"],
                "actions": len(actions),
                "action_generations": [generation for generation, _ in actions],
                "action_epochs": [action["atGlobalEpoch"] for _, action in actions],
                "action_labels": [action["shortLabel"] for _, action in actions],
                "epochs": epochs,
                "final_acc": epochs[-1]["valAcc"],
                "final_train_acc": epochs[-1]["trainAcc"],
                "best_acc": max(epoch["valAcc"] for epoch in epochs),
                "best_train_acc": max(epoch["trainAcc"] for epoch in epochs),
                "final_params": epochs[-1]["paramCount"],
                "start_params": epochs[0]["paramCount"],
            }
        )
    return runs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _completed(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [run for run in runs if run["status"] == "completed"]


def _generations(run: dict[str, object]) -> dict[int, list[dict[str, object]]]:
    generations: dict[int, list[dict[str, object]]] = defaultdict(list)
    for epoch in list(run["epochs"]):
        generations[int(epoch["generation"])].append(epoch)
    return generations


def _slope_angle(values: list[dict[str, object]]) -> float:
    slope = (float(values[-1]["trainAcc"]) - float(values[0]["trainAcc"])) / 2
    return degrees(atan(slope))


def _order_buckets(order_map: dict[int, list[float]]) -> list[list[float]]:
    return [
        order_map.get(0, []),
        order_map.get(1, []),
        order_map.get(2, []),
        order_map.get(3, []),
        [
            gain
            for order, gains in order_map.items()
            if order >= 4
            for gain in gains
        ],
    ]


def _plot_order_bars(
    axis: Axes,
    order_values: list[list[float]],
    bar_color: str,
    title: str,
) -> None:
    order_means = [_mean(values) * 100 if values else 0.0 for values in order_values]
    axis.bar(ORDER_LABELS, order_means, color=bar_color, alpha=0.35)
    for category, values in enumerate(order_values):
        count = len(values)
        if count == 0:
            continue
        offsets = (
            [0.0]
            if count == 1
            else [-0.16 + 0.32 * index / (count - 1) for index in range(count)]
        )
        axis.scatter(
            [category + offset for offset in offsets],
            [value * 100 for value in values],
            color="#222222",
            s=14,
            alpha=0.65,
        )
    axis.axhline(0, color="#222222", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Order of the action in one run")
    axis.grid(axis="y", alpha=0.25)


def _short_action_name(name: str) -> str:
    return name.replace(" Action", "").replace("Add ", "+").replace("Delete ", "−")


def generate_charts(
    output_dir: Path,
    runs_dir: Path = DEFAULT_RUNS,
    snapshot_path: Path | None = None,
) -> None:
    """Generate focused charts from Experiment 001 board output or snapshot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    if runs and snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    elif not runs and snapshot_path is not None and snapshot_path.exists():
        runs = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not runs:
        raise FileNotFoundError(f"No experiment runs or snapshot found for {runs_dir}")

    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    completed = _completed(runs)
    has_incomplete = len(completed) != len(runs)
    note_suffix = " · incomplete runs included" if has_incomplete else " · all 18 runs complete"
    positions = list(range(len(MODELS)))

    # Final accuracy by starting depth.
    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    for metric, offset, color, label in (
        ("final_train_acc", -0.18, "#3568a8", "Training accuracy"),
        ("final_acc", 0.18, "#4f8a63", "Validation accuracy"),
    ):
        means = [
            _mean([float(run[metric]) * 100 for run in completed if run["model"] == model])
            for model in MODELS
        ]
        bars = axis.bar(
            [position + offset for position in positions],
            means,
            width=0.36,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axis.set(
        title="Mean final training and validation accuracy by starting depth",
        xlabel="Starting model depth",
        ylabel="Mean final accuracy across slope thresholds and seeds (%)",
        xticks=positions,
        xticklabels=[MODEL_NAMES[model] for model in MODELS],
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, f"Source: full completed grid{note_suffix}", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-final-accuracy.png")
    plt.close(figure)

    # Final accuracy by slope threshold.
    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    angle_positions = list(range(len(ANGLES)))
    for metric, offset, color, label in (
        ("final_train_acc", -0.18, "#3568a8", "Training accuracy"),
        ("final_acc", 0.18, "#4f8a63", "Validation accuracy"),
    ):
        means = [
            _mean([float(run[metric]) * 100 for run in completed if run["angle"] == angle])
            for angle in ANGLES
        ]
        bars = axis.bar(
            [position + offset for position in angle_positions],
            means,
            width=0.36,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axis.set(
        title="Mean final training and validation accuracy by slope threshold",
        xlabel="Slope threshold",
        ylabel="Mean final accuracy across depths and seeds (%)",
        xticks=angle_positions,
        xticklabels=[f"{angle}°" for angle in ANGLES],
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, f"Source: full completed grid{note_suffix}", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-final-accuracy-by-slope.png")
    plt.close(figure)

    # Generation-0 capacity: one marker per seed. Slope does not change generation 0.
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    seed_offsets = {100: -0.12, 101: 0.12}
    for model_index, model in enumerate(MODELS):
        seed_values: dict[int, float] = {}
        for run in completed:
            if run["model"] != model:
                continue
            generations = _generations(run)
            seed_values[int(run["seed"])] = float(generations[0][-1]["valAcc"]) * 100
        if seed_values:
            axis.bar(
                model_index,
                _mean(list(seed_values.values())),
                color=MODEL_COLORS[model],
                alpha=0.28,
                width=0.55,
            )
        for seed, value in seed_values.items():
            axis.scatter(
                model_index + seed_offsets[seed],
                value,
                color=SEED_COLORS[seed],
                s=55,
                zorder=3,
                label=f"Seed {seed}" if model_index == 0 else None,
            )
    axis.set(
        title="Validation accuracy at the end of generation 0",
        xlabel="Starting model depth",
        ylabel="Validation accuracy (%)",
        xticks=positions,
        xticklabels=[MODEL_NAMES[model] for model in MODELS],
        ylim=(0, 40),
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys())
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        "Slope threshold does not change generation 0, so each seed appears once",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-generation-zero.png")
    plt.close(figure)

    # Parameter growth.
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for model_index, model in enumerate(MODELS):
        group = [run for run in completed if run["model"] == model]
        axis.bar(
            model_index - 0.18,
            _mean([float(run["start_params"]) for run in group]),
            width=0.36,
            color="#777777",
            label="Start parameters" if model_index == 0 else None,
        )
        axis.bar(
            model_index + 0.18,
            _mean([float(run["final_params"]) for run in group]),
            width=0.36,
            color=MODEL_COLORS[model],
            label="Final parameters" if model_index == 0 else None,
        )
        for run in group:
            axis.scatter(
                model_index + 0.18,
                float(run["final_params"]),
                color="#222222",
                s=18,
                zorder=3,
            )
    axis.set(
        title="Starting and final parameter counts",
        xlabel="Starting model depth",
        ylabel="Parameter count",
        xticks=positions,
        xticklabels=[MODEL_NAMES[model] for model in MODELS],
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, "Dots show final parameter counts for each completed run", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-param-growth.png")
    plt.close(figure)

    # Representative timeline: strongest final validation cell.
    representative = next(
        run
        for run in completed
        if run["angle"] == "2" and run["model"] == "big" and run["seed"] == 100
    )
    representative_epochs = list(representative["epochs"])
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(
        [int(epoch["globalEpoch"]) for epoch in representative_epochs],
        [float(epoch["trainAcc"]) * 100 for epoch in representative_epochs],
        color="#3568a8",
        label="Training accuracy",
    )
    lr_axis = axis.twinx()
    lr_axis.plot(
        [int(epoch["globalEpoch"]) for epoch in representative_epochs],
        [float(epoch["lr"]) for epoch in representative_epochs],
        color="#d18b2c",
        alpha=0.75,
        label="Learning rate",
    )
    for action_index, action_epoch in enumerate(list(representative["action_epochs"])):
        axis.axvline(
            int(action_epoch),
            color="#a65353",
            linestyle="--",
            linewidth=1,
            label="Architecture action" if action_index == 0 else None,
        )
    axis.set(
        title="Training accuracy and LR: 2° logistic, big starter, seed 100",
        xlabel="Global epoch",
        ylabel="Training accuracy (%)",
    )
    lr_axis.set_ylabel("Learning rate")
    handles, legend_labels = axis.get_legend_handles_labels()
    lr_handles, lr_labels = lr_axis.get_legend_handles_labels()
    axis.legend(handles + lr_handles, legend_labels + lr_labels, loc="lower right")
    axis.grid(alpha=0.2)
    figure.text(
        0.99,
        0.01,
        "Selected because it has the highest final validation accuracy in the full grid",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-representative-timeline.png")
    plt.close(figure)

    # Slope decisions: mean angle lines per threshold, plus overall mean.
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        by_angle_generation: dict[str, dict[int, list[float]]] = {
            angle: defaultdict(list) for angle in ANGLES
        }
        by_generation: dict[int, list[float]] = defaultdict(list)
        for run in completed:
            if run["model"] != model:
                continue
            generations = _generations(run)
            for generation in range(10):
                angle_value = _slope_angle(generations[generation])
                by_angle_generation[str(run["angle"])][generation].append(angle_value)
                by_generation[generation].append(angle_value)
        for angle in ANGLES:
            means = [
                _mean(by_angle_generation[angle][generation]) for generation in range(10)
            ]
            axis.plot(
                range(10),
                means,
                color=ANGLE_COLORS[angle],
                linewidth=1.2,
                label=f"{angle}° mean",
            )
        overall = [_mean(by_generation[generation]) for generation in range(10)]
        axis.plot(range(10), overall, color="#222222", linewidth=2.2, label="All-run mean")
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Generation")
        axis.set_xticks(list(range(10)))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Slope angle of training accuracy (degrees)")
    figure.suptitle("Mean slope angles by generation")
    figure.text(
        0.99,
        0.01,
        "Thin colored lines are means for each slope threshold; the thick black line is the mean over all runs",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    figure.savefig(output_dir / "001-slope-decisions.png")
    plt.close(figure)

    # Actions by generation and depth.
    figure, axis = plt.subplots(figsize=(10, 4.6))
    width = 0.25
    for model_index, model in enumerate(MODELS):
        counts = [
            sum(
                generation in list(run["action_generations"])
                for run in completed
                if run["model"] == model
            )
            for generation in range(10)
        ]
        axis.bar(
            [generation + (model_index - 1) * width for generation in range(10)],
            counts,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_NAMES[model],
        )
    axis.axvline(2.5, color="#777777", linestyle="--", linewidth=1)
    axis.set(
        title="Executed actions by generation and starting depth",
        xlabel="Generation",
        ylabel="Number of actions across six completed runs per depth",
        xticks=list(range(10)),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.text(
        0.99,
        0.01,
        "Dashed line separates early generations 0–2 from later generations · six runs per depth",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-actions-by-generation.png")
    plt.close(figure)

    # Recovery-window gains.
    action_gains_by_order: dict[int, list[float]] = defaultdict(list)
    action_gains_by_model_order: dict[str, dict[int, list[float]]] = {
        model: defaultdict(list) for model in MODELS
    }
    action_gains_by_angle_order: dict[str, dict[int, list[float]]] = {
        angle: defaultdict(list) for angle in ANGLES
    }
    effects_by_type: dict[str, list[float]] = defaultdict(list)
    train_effects_by_type: dict[str, list[float]] = defaultdict(list)
    effects_by_model_type: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in MODELS
    }
    train_effects_by_model_type: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in MODELS
    }
    disturbance_by_angle: dict[str, list[float]] = {angle: [] for angle in ANGLES}

    for run in completed:
        generations = _generations(run)
        action_map = dict(
            zip(list(run["action_generations"]), list(run["action_labels"]), strict=True)
        )
        model = str(run["model"])
        angle = str(run["angle"])
        action_order = 0
        for generation in range(1, max(generations) + 1):
            previous = generations[generation - 1]
            current = generations[generation]
            train_delta = float(current[0]["trainAcc"]) - float(previous[-1]["trainAcc"])
            if generation - 1 not in action_map:
                continue
            disturbance_by_angle[angle].append(train_delta)
            val_gain = float(current[-1]["valAcc"]) - float(previous[-1]["valAcc"])
            train_gain = float(current[-1]["trainAcc"]) - float(previous[-1]["trainAcc"])
            label = str(action_map[generation - 1])
            action_gains_by_order[action_order].append(val_gain)
            action_gains_by_model_order[model][action_order].append(val_gain)
            action_gains_by_angle_order[angle][action_order].append(val_gain)
            effects_by_type[label].append(val_gain)
            train_effects_by_type[label].append(train_gain)
            effects_by_model_type[model][label].append(val_gain)
            train_effects_by_model_type[model][label].append(train_gain)
            action_order += 1

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    _plot_order_bars(
        axis,
        _order_buckets(action_gains_by_order),
        "#3568a8",
        "Validation-accuracy change by action order",
    )
    axis.set_ylabel(
        "Validation-accuracy change over the next generation (percentage points)"
    )
    figure.text(
        0.99,
        0.01,
        "Bars are means; dots are individual observed actions from all completed runs",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-action-order.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        _plot_order_bars(
            axis,
            _order_buckets(action_gains_by_model_order[model]),
            MODEL_COLORS[model],
            MODEL_NAMES[model],
        )
    axes[0].set_ylabel(
        "Validation-accuracy change over the next generation (percentage points)"
    )
    figure.suptitle("Validation-accuracy change by action order and starting depth")
    figure.text(
        0.99,
        0.01,
        "Same recovery window as the pooled action-order chart · one panel per starting depth",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    figure.savefig(output_dir / "001-action-order-by-depth.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    for axis, angle in zip(axes, ANGLES, strict=True):
        _plot_order_bars(
            axis,
            _order_buckets(action_gains_by_angle_order[angle]),
            ANGLE_COLORS[angle],
            f"{angle}° slope threshold",
        )
    axes[0].set_ylabel(
        "Validation-accuracy change over the next generation (percentage points)"
    )
    figure.suptitle("Validation-accuracy change by action order and slope threshold")
    figure.text(
        0.99,
        0.01,
        "Same recovery window · one panel per slope threshold across all depths",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    figure.savefig(output_dir / "001-action-order-by-slope.png")
    plt.close(figure)

    # Action types pooled.
    type_names = sorted(effects_by_type)
    short_names = [_short_action_name(name) for name in type_names]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    type_positions = list(range(len(type_names)))
    axis.barh(
        [position - 0.18 for position in type_positions],
        [_mean(train_effects_by_type[name]) * 100 for name in type_names],
        height=0.34,
        color="#3568a8",
        alpha=0.4,
        label="Training accuracy",
    )
    axis.barh(
        [position + 0.18 for position in type_positions],
        [_mean(effects_by_type[name]) * 100 for name in type_names],
        height=0.34,
        color="#4f8a63",
        alpha=0.4,
        label="Validation accuracy",
    )
    for type_index, name in enumerate(type_names):
        for values, center, color in (
            (train_effects_by_type[name], type_index - 0.18, "#3568a8"),
            (effects_by_type[name], type_index + 0.18, "#4f8a63"),
        ):
            count = len(values)
            offsets = (
                [0.0]
                if count <= 1
                else [-0.1 + 0.2 * index / (count - 1) for index in range(count)]
            )
            axis.scatter(
                [value * 100 for value in values],
                [center + offset for offset in offsets],
                color=color,
                edgecolor="#222222",
                linewidth=0.3,
                s=15,
                alpha=0.75,
            )
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set(
        title="Training- and validation-accuracy change by action type",
        xlabel="Accuracy change over the next generation (percentage points)",
        yticks=type_positions,
        yticklabels=short_names,
    )
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        "Bars are means; colored dots are individual observed actions",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-action-types.png")
    plt.close(figure)

    # Action types by starting depth.
    all_type_names = sorted(
        {
            name
            for model in MODELS
            for name in effects_by_model_type[model]
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharex=True)
    for axis, model in zip(axes, MODELS, strict=True):
        short = [_short_action_name(name) for name in all_type_names]
        centers = list(range(len(all_type_names)))
        axis.barh(
            [center - 0.18 for center in centers],
            [_mean(train_effects_by_model_type[model][name]) * 100 for name in all_type_names],
            height=0.34,
            color="#3568a8",
            alpha=0.4,
            label="Training",
        )
        axis.barh(
            [center + 0.18 for center in centers],
            [_mean(effects_by_model_type[model][name]) * 100 for name in all_type_names],
            height=0.34,
            color="#4f8a63",
            alpha=0.4,
            label="Validation",
        )
        for type_index, name in enumerate(all_type_names):
            for values, center, color in (
                (train_effects_by_model_type[model][name], type_index - 0.18, "#3568a8"),
                (effects_by_model_type[model][name], type_index + 0.18, "#4f8a63"),
            ):
                if not values:
                    continue
                count = len(values)
                offsets = (
                    [0.0]
                    if count <= 1
                    else [-0.1 + 0.2 * index / (count - 1) for index in range(count)]
                )
                axis.scatter(
                    [value * 100 for value in values],
                    [center + offset for offset in offsets],
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.3,
                    s=14,
                    alpha=0.75,
                )
        axis.axvline(0, color="#222222", linewidth=1)
        axis.set_title(MODEL_NAMES[model])
        axis.set_yticks(centers)
        axis.set_yticklabels(short)
        axis.grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Accuracy change over the next generation (percentage points)")
    figure.suptitle("Accuracy change by action type and starting depth")
    figure.text(
        0.99,
        0.01,
        "Empty types mean that depth never executed that action in the completed grid",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    figure.savefig(output_dir / "001-action-types-by-depth.png")
    plt.close(figure)

    # Action composition counts by depth.
    figure, axis = plt.subplots(figsize=(10, 4.8))
    composition = {
        model: defaultdict(int)
        for model in MODELS
    }
    for run in completed:
        for label in list(run["action_labels"]):
            composition[str(run["model"])][str(label)] += 1
    type_list = sorted({name for model in MODELS for name in composition[model]})
    width = 0.25
    for model_index, model in enumerate(MODELS):
        counts = [composition[model][name] for name in type_list]
        axis.bar(
            [index + (model_index - 1) * width for index in range(len(type_list))],
            counts,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_NAMES[model],
        )
    axis.set(
        title="Executed action counts by type and starting depth",
        xlabel="Action type",
        ylabel="Number of executed actions across six runs",
        xticks=list(range(len(type_list))),
        xticklabels=[_short_action_name(name) for name in type_list],
    )
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        "Counts show what each starter actually searched, not only the mean accuracy effect",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_dir / "001-action-composition-by-depth.png")
    plt.close(figure)

    # Immediate disturbance by slope.
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    abs_means = [
        _mean([abs(value) for value in disturbance_by_angle[angle]]) * 100
        for angle in ANGLES
    ]
    signed_means = [_mean(disturbance_by_angle[angle]) * 100 for angle in ANGLES]
    bars = axis.bar(
        [f"{angle}°" for angle in ANGLES],
        abs_means,
        color=[ANGLE_COLORS[angle] for angle in ANGLES],
        alpha=0.45,
        label="Mean absolute change",
    )
    axis.plot(
        [f"{angle}°" for angle in ANGLES],
        signed_means,
        color="#222222",
        marker="o",
        linewidth=1.5,
        label="Mean signed change",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axis.set(
        title="Immediate training-accuracy change after actions by slope threshold",
        xlabel="Slope threshold",
        ylabel="Training-accuracy change (percentage points)",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        "Bars are absolute disturbance size; the line is signed direction after an action",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "001-generation-transition.png")
    plt.close(figure)

    # Training curves by depth.
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        for run in completed:
            if run["model"] != model:
                continue
            epochs = list(run["epochs"])
            axis.plot(
                [int(epoch["globalEpoch"]) for epoch in epochs],
                [float(epoch["trainAcc"]) * 100 for epoch in epochs],
                color=ANGLE_COLORS[str(run["angle"])],
                alpha=0.85,
                label=f"{run['angle']}° seed {run['seed']}",
            )
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=6)
    axes[0].set_ylabel("Training accuracy (%)")
    figure.suptitle("Training-accuracy curves by starting model depth")
    figure.text(0.99, 0.01, f"Source: board/metrics/training.json{note_suffix}", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    figure.savefig(output_dir / "001-training-curves.png")
    plt.close(figure)

    # Training curves by slope.
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for axis, angle in zip(axes, ANGLES, strict=True):
        for run in completed:
            if run["angle"] != angle:
                continue
            epochs = list(run["epochs"])
            axis.plot(
                [int(epoch["globalEpoch"]) for epoch in epochs],
                [float(epoch["trainAcc"]) * 100 for epoch in epochs],
                color=MODEL_COLORS[str(run["model"])],
                alpha=0.85,
                label=f"{MODEL_NAMES[str(run['model'])]} seed {run['seed']}",
            )
        axis.set_title(f"{angle}° slope threshold")
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=6)
    axes[0].set_ylabel("Training accuracy (%)")
    figure.suptitle("Training-accuracy curves by slope threshold")
    figure.text(0.99, 0.01, f"Source: board/metrics/training.json{note_suffix}", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 0.93))
    figure.savefig(output_dir / "001-training-curves-by-slope.png")
    plt.close(figure)


if __name__ == "__main__":
    generate_charts(DEFAULT_OUTPUT, snapshot_path=DEFAULT_SNAPSHOT)
