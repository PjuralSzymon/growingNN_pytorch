"""Generate charts for the MNIST slope-angle and LR-scheduler experiment."""

from __future__ import annotations

import json
from math import atan, cos, degrees, exp, pi, tanh
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITE = Path(__file__).parents[1]
DEFAULT_RUNS = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_mnist"
    / "runs"
    / "lr_scheduler_slope_angle_experiment"
)
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = (
    SITE
    / "data"
    / "experiments"
    / "experiment-000-slope-angle-lr-warmup.json"
)


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics and metadata for every slope-angle run."""
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
                "mode": parts[1].removeprefix("warmup_"),
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
            }
        )
    return runs


def generate_charts(
    output_dir: Path,
    runs_dir: Path = DEFAULT_RUNS,
    snapshot_path: Path | None = None,
) -> None:
    """Generate focused charts from the current experiment output."""
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
    angles = ("1", "3")
    modes = ("cosine", "logistic", "exponential")
    mode_names = {"cosine": "Cosine", "logistic": "Logistic", "exponential": "Exponential"}
    cells = [(angle, mode) for angle in angles for mode in modes]
    labels = [f"{angle}°\n{mode_names[mode]}" for angle, mode in cells]
    seed_colors = {1: "#3568a8", 2: "#4f8a63"}
    completed_count = sum(run["status"] == "completed" for run in runs)
    has_incomplete = completed_count != len(runs)

    figure, axis = plt.subplots(figsize=(10, 5.2))
    positions = list(range(len(cells)))
    for metric, offset, color, label in (
        ("final_train_acc", -0.18, "#3568a8", "Training accuracy"),
        ("final_acc", 0.18, "#4f8a63", "Validation accuracy"),
    ):
        means = []
        for cell in cells:
            values = [
                float(run[metric]) * 100
                for run in runs
                if (run["angle"], run["mode"]) == cell
            ]
            means.append(sum(values) / len(values) if values else 0.0)
        bars = axis.bar(
            [position + offset for position in positions],
            means,
            width=0.36,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axis.set(
        title="Mean final training and validation accuracy",
        xlabel="Slope threshold and LR warmup mode",
        ylabel="Mean final accuracy across two seeds (%)",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    accuracy_note = " · incomplete runs included" if has_incomplete else " · all runs complete"
    figure.text(0.99, 0.01, f"Source: final recorded epoch{accuracy_note}", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-final-accuracy.png")
    plt.close(figure)

    alpha = 0.01
    old_lr: list[float] = []
    old_epochs = list(range(33))
    for global_epoch in old_epochs:
        local_epoch = global_epoch % 11
        threshold = 2.0
        if local_epoch < threshold:
            value = alpha * (-(local_epoch - threshold) ** 2 / threshold**2 + 1)
        else:
            value = alpha * (-(local_epoch - threshold) ** 2 / (10 - threshold) ** 2 + 1)
        old_lr.append(max(0.001, value))
    new_epochs = list(range(33))
    new_lr: dict[str, list[float]] = {mode: [] for mode in modes}
    for global_epoch in new_epochs:
        since_change = global_epoch + 1 if global_epoch < 22 else global_epoch - 21
        x = max(0.0, min(since_change / 10, 1.0))
        new_lr["cosine"].append(max(0.001, alpha * (1 - cos(pi * x)) / 2))
        low = (1 + tanh(-10 / 4)) / 2
        high = (1 + tanh(10 / 4)) / 2
        logistic = alpha * (((1 + tanh(10 * (x - 0.5) / 2)) / 2) - low) / (high - low)
        new_lr["logistic"].append(max(0.001, logistic))
        exponential = alpha * (1 - exp(-10 * x)) / (1 - exp(-10))
        new_lr["exponential"].append(max(0.001, exponential))
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    axes[0].plot(old_epochs, old_lr, color="#a65353", linewidth=2)
    axes[0].set(title="Previous generation-cyclic schedule", ylabel="Learning rate")
    axes[0].grid(alpha=0.25)
    for boundary in (11, 22):
        axes[0].axvline(boundary, color="#777777", linestyle=":", linewidth=1)
        axes[0].scatter(boundary, old_lr[boundary], color="#d62728", marker="X", s=65, zorder=3)
        axes[0].annotate(
            "Action at low LR",
            (boundary, old_lr[boundary]),
            xytext=(boundary - 4, old_lr[boundary] + 0.002),
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=7,
        )
    mode_colors = {"cosine": "#3568a8", "logistic": "#4f8a63", "exponential": "#d18b2c"}
    for mode in modes:
        axes[1].plot(new_epochs, new_lr[mode], color=mode_colors[mode], label=mode_names[mode], linewidth=2)
    axes[1].axvline(22, color="#222222", linestyle="--", linewidth=1, label="Structure changed")
    axes[1].set(title="Action-aware warmup schedule", xlabel="Global epoch", ylabel="Learning rate")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.text(0.99, 0.01, "Conceptual comparison · old schedule cycles each generation; new schedule resets only after an action", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-learning-rate-design.png")
    plt.close(figure)

    conceptual_epochs = list(range(41))
    stable_accuracy = [35 + 1.4 * epoch - 0.013 * epoch**2 for epoch in conceptual_epochs]
    risk_accuracy = stable_accuracy[:21]
    for epoch in range(21, 41):
        recovery = stable_accuracy[epoch] - 18 * exp(-(epoch - 21) / 5)
        oscillation = 5 * exp(-(epoch - 21) / 6) * cos((epoch - 21) * pi / 2)
        risk_accuracy.append(recovery + oscillation)
    figure, axis = plt.subplots(figsize=(9.5, 4.2))
    axis.plot(conceptual_epochs, stable_accuracy, color="#4f8a63", linewidth=2, label="Desired stable training")
    axis.plot(conceptual_epochs, risk_accuracy, color="#a65353", linewidth=2, label="Possible action shock")
    axis.axvline(20, color="#222222", linestyle="--", linewidth=1)
    axis.annotate("Architecture action", (20, risk_accuracy[20]), xytext=(12, 78), arrowprops={"arrowstyle": "->"})
    axis.set(
        title="Why LR warmup follows an architecture change",
        xlabel="Training epoch",
        ylabel="Conceptual training accuracy (%)",
        ylim=(25, 90),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.text(0.99, 0.01, "Conceptual drawing, not measured experiment data", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-instability-risk.png")
    plt.close(figure)

    sorted_runs = sorted(runs, key=lambda run: (int(run["angle"]), modes.index(str(run["mode"])), int(run["seed"])))
    representative = next(
        run for run in sorted_runs
        if run["angle"] == "3" and run["mode"] == "logistic" and run["seed"] == 1
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
        title="Training accuracy and LR: 3° logistic, seed 1",
        xlabel="Global epoch",
        ylabel="Training accuracy (%)",
    )
    lr_axis.set_ylabel("Learning rate")
    handles, legend_labels = axis.get_legend_handles_labels()
    lr_handles, lr_labels = lr_axis.get_legend_handles_labels()
    axis.legend(handles + lr_handles, legend_labels + lr_labels, loc="lower right")
    axis.grid(alpha=0.2)
    figure.text(0.99, 0.01, "Red dashed lines mark executed architecture actions", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-representative-timeline.png")
    plt.close(figure)

    representative_generations: dict[int, list[dict[str, object]]] = {}
    for epoch in representative_epochs:
        representative_generations.setdefault(int(epoch["generation"]), []).append(epoch)
    decision_angles = []
    for generation in range(10):
        values = representative_generations[generation]
        slope = (float(values[-1]["trainAcc"]) - float(values[0]["trainAcc"])) / 2
        decision_angles.append(degrees(atan(slope)))
    threshold = float(representative["angle"])
    decision_colors = ["#4f8a63" if abs(angle) <= threshold else "#9aa4ad" for angle in decision_angles]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    bars = axis.bar(range(10), decision_angles, color=decision_colors)
    axis.axhspan(-threshold, threshold, color="#4f8a63", alpha=0.12, label="Simulation zone: |angle| ≤ 3°")
    axis.axhline(threshold, color="#4f8a63", linestyle="--", linewidth=1)
    axis.axhline(-threshold, color="#4f8a63", linestyle="--", linewidth=1)
    for generation, (bar, angle) in enumerate(zip(bars, decision_angles, strict=True)):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            angle + (0.25 if angle >= 0 else -0.45),
            f"{angle:.2f}°",
            ha="center",
            va="bottom" if angle >= 0 else "top",
            fontsize=7,
        )
        if generation in list(representative["action_generations"]):
            axis.text(generation, 0, "A", ha="center", va="center", color="#8b1a1a", fontweight="bold")
    axis.set(
        title="Slope decision by generation: 3° logistic, seed 1",
        xlabel="Generation",
        ylabel="Signed training-accuracy slope angle (degrees)",
        xticks=list(range(10)),
    )
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.2)
    figure.text(0.99, 0.01, "A marks an executed action; gray bars are outside the simulation zone", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-slope-decisions.png")
    plt.close(figure)

    actions_by_threshold = {
        angle: [
            sum(
                generation in list(run["action_generations"])
                for run in runs
                if run["angle"] == angle
            )
            for generation in range(10)
        ]
        for angle in angles
    }
    figure, axis = plt.subplots(figsize=(9.5, 4.2))
    axis.bar(
        range(10),
        actions_by_threshold["1"],
        color="#3568a8",
        label="1° threshold",
    )
    axis.bar(
        range(10),
        actions_by_threshold["3"],
        bottom=actions_by_threshold["1"],
        color="#4f8a63",
        label="3° threshold",
    )
    axis.set(
        title="Executed actions by generation",
        xlabel="Generation",
        ylabel="Number of actions across 12 runs",
        xticks=list(range(10)),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.text(0.99, 0.01, "Generation 9 actions have no later training or evaluation generation", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-actions-by-generation.png")
    plt.close(figure)

    no_action_boundaries: list[tuple[float, float]] = []
    action_boundaries_by_mode: dict[str, list[tuple[float, float]]] = {}
    first_action_gain: list[float] = []
    later_action_gain: list[float] = []
    action_gains_by_order: dict[int, list[float]] = {}
    effects_by_type: dict[str, list[float]] = {}
    train_effects_by_type: dict[str, list[float]] = {}
    for run in runs:
        generations: dict[int, list[dict[str, object]]] = {}
        for epoch in list(run["epochs"]):
            generations.setdefault(int(epoch["generation"]), []).append(epoch)
        action_map = dict(zip(list(run["action_generations"]), list(run["action_labels"]), strict=True))
        action_order = 0
        for generation in range(1, 10):
            previous = generations[generation - 1]
            current = generations[generation]
            boundary = (
                float(current[0]["trainAcc"]) - float(previous[-1]["trainAcc"]),
                float(current[0]["valAcc"]) - float(previous[-1]["valAcc"]),
            )
            if generation - 1 not in action_map:
                no_action_boundaries.append(boundary)
                continue
            action_boundaries_by_mode.setdefault(str(run["mode"]), []).append(boundary)
            gain = float(current[-1]["valAcc"]) - float(previous[-1]["valAcc"])
            train_gain = float(current[-1]["trainAcc"]) - float(previous[-1]["trainAcc"])
            if action_order == 0:
                first_action_gain.append(gain)
            else:
                later_action_gain.append(gain)
            action_gains_by_order.setdefault(action_order, []).append(gain)
            action_order += 1
            effects_by_type.setdefault(action_map[generation - 1], []).append(gain)
            train_effects_by_type.setdefault(action_map[generation - 1], []).append(train_gain)
    transition_labels = ["No action", "Cosine\nafter action", "Logistic\nafter action", "Exponential\nafter action"]
    transition_groups = [
        no_action_boundaries,
        *(action_boundaries_by_mode.get(mode, []) for mode in modes),
    ]
    transition_colors = ["#777777", "#3568a8", "#4f8a63", "#d18b2c"]
    absolute_means = [
        sum(abs(item[0]) for item in group) / len(group) * 100 if group else 0.0
        for group in transition_groups
    ]
    figure, axis = plt.subplots(figsize=(9, 4.4))
    bars = axis.bar(
        transition_labels,
        absolute_means,
        color=transition_colors,
    )
    axis.bar_label(bars, fmt="%.2f percentage points", fontsize=8, padding=3)
    axis.set(
        title="Mean absolute training-accuracy change between generations",
        ylabel="Mean absolute accuracy change (percentage points)",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, "Absolute change measures disturbance size, not improvement", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-generation-transition.png")
    plt.close(figure)

    signed_means = [
        sum(item[0] for item in group) / len(group) * 100 if group else 0.0
        for group in transition_groups
    ]
    drop_counts = [sum(item[0] < 0 for item in group) for group in transition_groups]
    figure, axis = plt.subplots(figsize=(9, 4.6))
    bars = axis.bar(
        transition_labels,
        signed_means,
        color=transition_colors,
    )
    for bar, mean, drops, group in zip(
        bars, signed_means, drop_counts, transition_groups, strict=True
    ):
        axis.annotate(
            f"{mean:+.2f} percentage points\n{drops}/{len(group)} drops",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4 if mean >= 0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=8,
        )
    axis.axhline(0, color="#222222", linewidth=1)
    axis.set(
        title="Mean signed training-accuracy change between generations",
        ylabel="Mean accuracy change (percentage points)",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, "Negative values and drop counts show harmful immediate changes", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-signed-generation-transition.png")
    plt.close(figure)

    order_labels = ["1st", "2nd", "3rd", "4th", "5th+"]
    order_values = [
        action_gains_by_order.get(0, []),
        action_gains_by_order.get(1, []),
        action_gains_by_order.get(2, []),
        action_gains_by_order.get(3, []),
        [
            gain
            for order, gains in action_gains_by_order.items()
            if order >= 4
            for gain in gains
        ],
    ]
    order_means = [sum(values) / len(values) * 100 if values else 0.0 for values in order_values]
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.bar(order_labels, order_means, color=["#3568a8", "#777777", "#777777", "#777777", "#a65353"], alpha=0.35)
    for category, values in enumerate(order_values):
        count = len(values)
        offsets = [0.0] if count == 1 else [-0.16 + 0.32 * index / (count - 1) for index in range(count)]
        axis.scatter(
            [category + offset for offset in offsets],
            [value * 100 for value in values],
            color="#222222",
            s=14,
            alpha=0.65,
        )
    axis.axhline(0, color="#222222", linewidth=1)
    axis.set(
        title="Validation-accuracy change by action order",
        xlabel="Order of the action in one run",
        ylabel="Validation-accuracy change over the next generation (percentage points)",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, "Bars are means; dots are individual observed actions", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-action-order.png")
    plt.close(figure)

    type_names = sorted(effects_by_type)
    short_names = [name.replace(" Action", "").replace("Add ", "+").replace("Delete ", "−") for name in type_names]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    type_positions = list(range(len(type_names)))
    axis.barh(
        [position - 0.18 for position in type_positions],
        [sum(train_effects_by_type[name]) / len(train_effects_by_type[name]) * 100 for name in type_names],
        height=0.34,
        color="#3568a8",
        alpha=0.4,
        label="Training accuracy",
    )
    axis.barh(
        [position + 0.18 for position in type_positions],
        [sum(effects_by_type[name]) / len(effects_by_type[name]) * 100 for name in type_names],
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
            offsets = [0.0] if count == 1 else [-0.1 + 0.2 * index / (count - 1) for index in range(count)]
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
    figure.text(0.99, 0.01, "Bars are means; colored dots are individual observed actions", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output_dir / "000-action-types.png")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    for row_index, angle in enumerate(angles):
        for column_index, mode in enumerate(modes):
            axis = axes[row_index][column_index]
            for run in runs:
                if run["angle"] != angle or run["mode"] != mode:
                    continue
                epochs = list(run["epochs"])
                axis.plot(
                    [int(epoch["globalEpoch"]) for epoch in epochs],
                    [float(epoch["trainAcc"]) * 100 for epoch in epochs],
                    color=seed_colors[int(run["seed"])],
                    linestyle="--" if run["status"] != "completed" else "-",
                    label=f"Seed {run['seed']}",
                )
            axis.set_title(f"{angle}° · {mode_names[mode]}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("Global epoch")
            if column_index == 0:
                axis.set_ylabel("Training accuracy (%)")
            handles, legend_labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, legend_labels, fontsize=7)
    figure.suptitle("Training-accuracy curves for the full slope-angle warmup grid")
    curve_note = " · dashed lines are incomplete" if has_incomplete else " · all runs complete"
    figure.text(0.99, 0.01, f"Source: board/metrics/training.json{curve_note}", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(output_dir / "000-training-curves.png")
    plt.close(figure)


if __name__ == "__main__":
    generate_charts(DEFAULT_OUTPUT, snapshot_path=DEFAULT_SNAPSHOT)
