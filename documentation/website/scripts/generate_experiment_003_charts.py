"""Generate charts for Experiment 003: simulation score accuracy metric.

Loads two grids:
- before_fix: exp003_score_accuracy_metric (stacked dropout on path possible)
- after_fix: exp003_score_accuracy_metric_after_fix_1 (path ban)

Writes per-phase charts (`003-before-*`, `003-after-*`) and compare charts when both exist.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITE = Path(__file__).parents[1]
_RUNS_ROOT = SITE.parents[1] / "experiments" / "output" / "train_mnist" / "runs"
DEFAULT_BEFORE_RUNS = _RUNS_ROOT / "exp003_score_accuracy_metric"
DEFAULT_AFTER_RUNS = _RUNS_ROOT / "exp003_score_accuracy_metric_after_fix_1"
# Backward-compatible alias used by older tests/callers.
DEFAULT_RUNS = DEFAULT_BEFORE_RUNS
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-003-score-accuracy-metric.json"

PHASE_BEFORE = "before_fix"
PHASE_AFTER = "after_fix"
PHASE_LABELS = {
    PHASE_BEFORE: "before fix",
    PHASE_AFTER: "after fix",
}
PHASE_PREFIXES = {
    PHASE_BEFORE: "003-before",
    PHASE_AFTER: "003-after",
}

SCORE_METRICS = ("val_acc", "train_acc")
MODELS = ("big", "medium_1conv_2linear")
MODEL_COLORS = {
    "big": "#3568a8",
    "medium_1conv_2linear": "#4f8a63",
}
SCORE_COLORS = {
    "val_acc": "#3568a8",
    "train_acc": "#d18b2c",
}
PHASE_COLORS = {
    PHASE_BEFORE: "#8a4f3d",
    PHASE_AFTER: "#3568a8",
}
SEED_COLORS = {100: "#3568a8", 101: "#4f8a63", 102: "#d18b2c", 103: "#7a5a9a"}
SHORT_NAMES = {
    "medium_1conv_2linear": "med 1c+2l",
    "val_acc": "grade val",
    "train_acc": "grade train",
}

PHASE_CHART_STEMS = (
    "final-accuracy-by-score-metric",
    "grading-overall-final-validation",
    "grading-by-model-final-validation",
    "dropout-actions-by-score-metric",
    "action-composition-by-score-metric",
    "action-types",
    "action-types-by-score-metric",
    "training-curves",
)


def _short(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _short_action_name(name: str) -> str:
    return name.replace(" Action", "").replace("Add ", "+").replace("Delete ", "−")


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics for every score-metric × model × seed run."""
    runs: list[dict[str, object]] = []
    if not runs_dir.exists():
        return runs
    for main_path in sorted(runs_dir.rglob("board/main.json")):
        run_dir = main_path.parent.parent
        parts = run_dir.relative_to(runs_dir).parts
        metrics_path = main_path.parent / "metrics" / "training.json"
        if not metrics_path.exists() or len(parts) < 2:
            continue
        main = json.loads(main_path.read_text(encoding="utf-8"))
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        actions = [
            (item["generation"], item["actionExecuted"])
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        labels = [action["shortLabel"] for _, action in actions]
        runs.append(
            {
                "score_metric": parts[0],
                "model": parts[1],
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "elapsed_sec": main.get("trainingTimeElapsedSec"),
                "started_on": main.get("experimentStartedOn"),
                "last_update": main.get("lastUpdate"),
                "actions": len(actions),
                "action_generations": [generation for generation, _ in actions],
                "action_labels": labels,
                "dropout_actions": sum(
                    1 for label in labels if "Dropout" in str(label)
                ),
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


def _compact_run(run: dict[str, object]) -> dict[str, object]:
    return {
        "score_metric": run["score_metric"],
        "model": run["model"],
        "seed": run["seed"],
        "status": run["status"],
        "elapsed_sec": run["elapsed_sec"],
        "started_on": run.get("started_on"),
        "last_update": run.get("last_update"),
        "final_acc": run["final_acc"],
        "final_train_acc": run["final_train_acc"],
        "best_acc": run["best_acc"],
        "best_train_acc": run["best_train_acc"],
        "final_params": run["final_params"],
        "start_params": run["start_params"],
        "actions": run["actions"],
        "action_generations": run["action_generations"],
        "action_labels": run["action_labels"],
        "dropout_actions": run["dropout_actions"],
        "epochs": [
            {
                "globalEpoch": epoch["globalEpoch"],
                "generation": epoch.get("generation"),
                "trainAcc": epoch["trainAcc"],
                "valAcc": epoch["valAcc"],
                "lr": epoch.get("lr"),
                "paramCount": epoch["paramCount"],
            }
            for epoch in run["epochs"]  # type: ignore[union-attr]
        ],
    }


def write_snapshot(
    phases: dict[str, list[dict[str, object]]],
    snapshot_path: Path,
    folders: dict[str, str] | None = None,
) -> None:
    """Persist before/after run snapshots for documentation without raw output."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "folders": folders
        or {
            PHASE_BEFORE: DEFAULT_BEFORE_RUNS.name,
            PHASE_AFTER: DEFAULT_AFTER_RUNS.name,
        }
    }
    for phase, runs in phases.items():
        payload[phase] = {"runs": [_compact_run(run) for run in runs]}
    # Keep legacy top-level runs = before_fix for older readers.
    before_runs = phases.get(PHASE_BEFORE, [])
    payload["runs"] = [_compact_run(run) for run in before_runs]
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_phases_or_snapshot(
    before_runs_dir: Path,
    after_runs_dir: Path,
    snapshot_path: Path,
) -> dict[str, list[dict[str, object]]]:
    """Prefer raw before/after folders; fall back to the committed snapshot."""
    before_raw = load_runs(before_runs_dir)
    after_raw = load_runs(after_runs_dir)
    if before_raw or after_raw:
        return {
            PHASE_BEFORE: before_raw,
            PHASE_AFTER: after_raw,
        }
    if not snapshot_path.exists():
        return {PHASE_BEFORE: [], PHASE_AFTER: []}
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    before = list(payload.get(PHASE_BEFORE, {}).get("runs", payload.get("runs", [])))
    after = list(payload.get(PHASE_AFTER, {}).get("runs", []))
    return {PHASE_BEFORE: before, PHASE_AFTER: after}


def _write_phase_charts(
    completed: list[dict[str, object]],
    output_dir: Path,
    prefix: str,
    note: str,
    save: Callable[[plt.Figure, str], Path],
) -> list[Path]:
    """Write the standard Exp 003 analysis charts for one phase."""
    if not completed:
        return []
    written: list[Path] = []

    # Final accuracy: score metric × model.
    labels = []
    positions = []
    index = 0
    for score_metric in SCORE_METRICS:
        for model in MODELS:
            labels.append(f"{_short(score_metric)}\n{_short(model)}")
            positions.append(index)
            index += 1

    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    for metric, offset, color, label in (
        ("final_train_acc", -0.18, "#3568a8", "Training accuracy"),
        ("final_acc", 0.18, "#4f8a63", "Validation accuracy"),
    ):
        means = []
        for score_metric in SCORE_METRICS:
            for model in MODELS:
                group = [
                    run
                    for run in completed
                    if run["score_metric"] == score_metric and run["model"] == model
                ]
                means.append(_mean([float(run[metric]) * 100 for run in group]))
        bars = axis.bar(
            [position + offset for position in positions],
            means,
            width=0.36,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    for position, (score_metric, model) in enumerate(
        (score_metric, model)
        for score_metric in SCORE_METRICS
        for model in MODELS
    ):
        for run in completed:
            if run["score_metric"] != score_metric or run["model"] != model:
                continue
            axis.scatter(
                position - 0.18,
                float(run["final_train_acc"]) * 100,
                color="#1f3f6d",
                s=18,
                zorder=3,
            )
            axis.scatter(
                position + 0.18,
                float(run["final_acc"]) * 100,
                color="#222222",
                s=18,
                zorder=3,
            )
    axis.set(
        title="Mean final accuracy by grading mode and starter",
        xlabel="Grading mode × starter",
        ylabel="Mean final accuracy (%)",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · dots are per-seed finals (blue=train, black=val)",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, f"{prefix}-final-accuracy-by-score-metric.png"))

    # Clear answer chart: validation grading vs training grading overall.
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    overall_means = []
    overall_labels = []
    for score_metric in SCORE_METRICS:
        group = [run for run in completed if run["score_metric"] == score_metric]
        overall_means.append(_mean([float(run["final_acc"]) * 100 for run in group]))
        overall_labels.append(_short(score_metric))
    bars = axis.bar(
        overall_labels,
        overall_means,
        color=[SCORE_COLORS[score_metric] for score_metric in SCORE_METRICS],
        width=0.55,
    )
    axis.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    for score_index, score_metric in enumerate(SCORE_METRICS):
        for run in completed:
            if run["score_metric"] != score_metric:
                continue
            axis.scatter(
                score_index,
                float(run["final_acc"]) * 100,
                color="#222222",
                s=18,
                zorder=3,
            )
    winner = (
        SCORE_METRICS[0]
        if overall_means[0] >= overall_means[1]
        else SCORE_METRICS[1]
    )
    axis.set(
        title=(
            f"Overall mean final validation: "
            f"{_short(winner)} wins this phase"
        ),
        xlabel="Simulation grading metric",
        ylabel="Mean final validation accuracy (%)",
        ylim=(0, 100),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · each bar pools both starters · dots are seeds",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, f"{prefix}-grading-overall-final-validation.png"))

    # Same answer split by starter.
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    width = 0.36
    model_positions = list(range(len(MODELS)))
    for metric_index, score_metric in enumerate(SCORE_METRICS):
        means = []
        for model in MODELS:
            group = [
                run
                for run in completed
                if run["score_metric"] == score_metric and run["model"] == model
            ]
            means.append(_mean([float(run["final_acc"]) * 100 for run in group]))
        xs = [position + (metric_index - 0.5) * width for position in model_positions]
        bars = axis.bar(
            xs,
            means,
            width=width,
            color=SCORE_COLORS[score_metric],
            label=_short(score_metric),
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        for model_index, model in enumerate(MODELS):
            for run in completed:
                if run["score_metric"] != score_metric or run["model"] != model:
                    continue
                axis.scatter(
                    model_index + (metric_index - 0.5) * width,
                    float(run["final_acc"]) * 100,
                    color="#222222",
                    s=16,
                    zorder=3,
                )
    axis.set(
        title="Mean final validation by starter: grade val vs grade train",
        xlabel="Starter",
        ylabel="Mean final validation accuracy (%)",
        xticks=model_positions,
        xticklabels=[_short(model) for model in MODELS],
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, f"{note} · dots are per-seed finals", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, f"{prefix}-grading-by-model-final-validation.png"))

    # Dropout counts.
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    width = 0.36
    for metric_index, score_metric in enumerate(SCORE_METRICS):
        means = []
        for model in MODELS:
            group = [
                run
                for run in completed
                if run["score_metric"] == score_metric and run["model"] == model
            ]
            means.append(_mean([float(run["dropout_actions"]) for run in group]))
        xs = [index + (metric_index - 0.5) * width for index in range(len(MODELS))]
        bars = axis.bar(
            xs,
            means,
            width=width,
            color=SCORE_COLORS[score_metric],
            label=_short(score_metric),
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        for model_index, model in enumerate(MODELS):
            for run in completed:
                if run["score_metric"] != score_metric or run["model"] != model:
                    continue
                axis.scatter(
                    model_index + (metric_index - 0.5) * width,
                    float(run["dropout_actions"]),
                    color="#222222",
                    s=18,
                    zorder=3,
                )
    axis.set(
        title="Mean sequential-dropout actions by grading mode",
        xlabel="Starter",
        ylabel="Mean dropout actions per completed seed",
        xticks=list(range(len(MODELS))),
        xticklabels=[_short(model) for model in MODELS],
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, note, ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, f"{prefix}-dropout-actions-by-score-metric.png"))

    # Action composition totals by grading mode.
    composition: dict[str, dict[str, int]] = {
        score_metric: defaultdict(int) for score_metric in SCORE_METRICS
    }
    for run in completed:
        for label in list(run["action_labels"]):
            composition[str(run["score_metric"])][str(label)] += 1
    type_list = sorted(
        {name for score_metric in SCORE_METRICS for name in composition[score_metric]}
    )
    if type_list:
        figure, axis = plt.subplots(figsize=(10.5, 4.8))
        width = 0.36
        short_types = [_short_action_name(name) for name in type_list]
        for metric_index, score_metric in enumerate(SCORE_METRICS):
            counts = [composition[score_metric][name] for name in type_list]
            xs = [index + (metric_index - 0.5) * width for index in range(len(type_list))]
            axis.bar(
                xs,
                counts,
                width=width,
                color=SCORE_COLORS[score_metric],
                label=_short(score_metric),
            )
        axis.set(
            title="Executed action counts by type and grading mode",
            xlabel="Action type",
            ylabel="Number of executed actions across completed seeds",
            xticks=list(range(len(type_list))),
            xticklabels=short_types,
        )
        axis.tick_params(axis="x", rotation=20)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.text(0.99, 0.01, note, ha="right", fontsize=7)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        written.append(save(figure, f"{prefix}-action-composition-by-score-metric.png"))

    # Recovery-window gains by action type.
    train_effects_by_type: dict[str, list[float]] = defaultdict(list)
    val_effects_by_type: dict[str, list[float]] = defaultdict(list)
    train_effects_by_metric_type: dict[str, dict[str, list[float]]] = {
        score_metric: defaultdict(list) for score_metric in SCORE_METRICS
    }
    val_effects_by_metric_type: dict[str, dict[str, list[float]]] = {
        score_metric: defaultdict(list) for score_metric in SCORE_METRICS
    }
    for run in completed:
        epochs = list(run["epochs"])
        by_generation: dict[int, list[dict[str, object]]] = defaultdict(list)
        for epoch in epochs:
            by_generation[int(epoch["generation"])].append(epoch)
        action_map = dict(
            zip(list(run["action_generations"]), list(run["action_labels"]), strict=True)
        )
        score_metric = str(run["score_metric"])
        for generation, label in action_map.items():
            previous = by_generation.get(int(generation))
            current = by_generation.get(int(generation) + 1)
            if not previous or not current:
                continue
            train_gain = float(current[-1]["trainAcc"]) - float(previous[-1]["trainAcc"])
            val_gain = float(current[-1]["valAcc"]) - float(previous[-1]["valAcc"])
            train_effects_by_type[str(label)].append(train_gain)
            val_effects_by_type[str(label)].append(val_gain)
            train_effects_by_metric_type[score_metric][str(label)].append(train_gain)
            val_effects_by_metric_type[score_metric][str(label)].append(val_gain)

    type_names = sorted(
        train_effects_by_type, key=lambda name: -len(train_effects_by_type[name])
    )
    if type_names:
        short_names = [_short_action_name(name) for name in type_names]
        figure, axis = plt.subplots(figsize=(9.8, 5.0))
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
            [_mean(val_effects_by_type[name]) * 100 for name in type_names],
            height=0.34,
            color="#4f8a63",
            alpha=0.4,
            label="Validation accuracy",
        )
        for type_index, name in enumerate(type_names):
            for values, center, color in (
                (train_effects_by_type[name], type_index - 0.18, "#3568a8"),
                (val_effects_by_type[name], type_index + 0.18, "#4f8a63"),
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
            (
                f"{note} · pooled across both grading modes · "
                "bars are means; dots are individual actions"
            ),
            ha="right",
            fontsize=7,
        )
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        written.append(save(figure, f"{prefix}-action-types.png"))

        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
        for axis, score_metric in zip(axes, SCORE_METRICS, strict=True):
            metric_types = sorted(
                train_effects_by_metric_type[score_metric],
                key=lambda name: -len(train_effects_by_metric_type[score_metric][name]),
            )
            if not metric_types:
                axis.set_title(_short(score_metric))
                continue
            positions = list(range(len(metric_types)))
            axis.barh(
                [position - 0.18 for position in positions],
                [
                    _mean(train_effects_by_metric_type[score_metric][name]) * 100
                    for name in metric_types
                ],
                height=0.34,
                color="#3568a8",
                alpha=0.4,
                label="Training",
            )
            axis.barh(
                [position + 0.18 for position in positions],
                [
                    _mean(val_effects_by_metric_type[score_metric][name]) * 100
                    for name in metric_types
                ],
                height=0.34,
                color="#4f8a63",
                alpha=0.4,
                label="Validation",
            )
            for type_index, name in enumerate(metric_types):
                for values, center, color in (
                    (
                        train_effects_by_metric_type[score_metric][name],
                        type_index - 0.18,
                        "#3568a8",
                    ),
                    (
                        val_effects_by_metric_type[score_metric][name],
                        type_index + 0.18,
                        "#4f8a63",
                    ),
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
                        s=14,
                        alpha=0.75,
                    )
            axis.axvline(0, color="#222222", linewidth=1)
            axis.set_title(_short(score_metric))
            axis.set_yticks(positions)
            axis.set_yticklabels([_short_action_name(name) for name in metric_types])
            axis.grid(axis="x", alpha=0.25)
            axis.legend(fontsize=7)
        axes[0].set_xlabel("Accuracy change (percentage points)")
        axes[1].set_xlabel("Accuracy change (percentage points)")
        figure.suptitle("Action-type accuracy change by grading mode")
        figure.text(0.99, 0.01, note, ha="right", fontsize=7)
        figure.tight_layout(rect=(0, 0.03, 1, 0.93))
        written.append(save(figure, f"{prefix}-action-types-by-score-metric.png"))

    # Training curves.
    combos = [
        (score_metric, model) for score_metric in SCORE_METRICS for model in MODELS
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharey=True)
    flat = list(axes.flat)
    for axis, (score_metric, model) in zip(flat, combos, strict=True):
        for run in completed:
            if run["score_metric"] != score_metric or run["model"] != model:
                continue
            epochs = list(run["epochs"])
            axis.plot(
                [int(epoch["globalEpoch"]) for epoch in epochs],
                [float(epoch["trainAcc"]) * 100 for epoch in epochs],
                color=SEED_COLORS.get(int(run["seed"]), "#3568a8"),
                alpha=0.85,
                label=f"seed {run['seed']}",
            )
        axis.set_title(f"{_short(score_metric)} · {_short(model)}")
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=6)
    flat[0].set_ylabel("Training accuracy (%)")
    flat[2].set_ylabel("Training accuracy (%)")
    figure.suptitle("Training-accuracy curves by grading mode and starter")
    figure.text(0.99, 0.01, note, ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    written.append(save(figure, f"{prefix}-training-curves.png"))

    return written


def _write_compare_charts(
    before: list[dict[str, object]],
    after: list[dict[str, object]],
    note: str,
    save: Callable[[plt.Figure, str], Path],
) -> list[Path]:
    """Write before-vs-after comparison charts for matched score × model cells."""
    written: list[Path] = []
    labels = []
    positions = []
    for index, (score_metric, model) in enumerate(
        (score_metric, model) for score_metric in SCORE_METRICS for model in MODELS
    ):
        labels.append(f"{_short(score_metric)}\n{_short(model)}")
        positions.append(index)

    # Mean final validation before vs after.
    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    width = 0.36
    for phase_index, (phase_runs, phase_key) in enumerate(
        ((before, PHASE_BEFORE), (after, PHASE_AFTER))
    ):
        means = []
        for score_metric in SCORE_METRICS:
            for model in MODELS:
                group = [
                    run
                    for run in phase_runs
                    if run["status"] == "completed"
                    and run["score_metric"] == score_metric
                    and run["model"] == model
                ]
                means.append(_mean([float(run["final_acc"]) * 100 for run in group]))
        xs = [position + (phase_index - 0.5) * width for position in positions]
        bars = axis.bar(
            xs,
            means,
            width=width,
            color=PHASE_COLORS[phase_key],
            label=PHASE_LABELS[phase_key],
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        for position, (score_metric, model) in enumerate(
            (score_metric, model)
            for score_metric in SCORE_METRICS
            for model in MODELS
        ):
            for run in phase_runs:
                if (
                    run["status"] != "completed"
                    or run["score_metric"] != score_metric
                    or run["model"] != model
                ):
                    continue
                axis.scatter(
                    position + (phase_index - 0.5) * width,
                    float(run["final_acc"]) * 100,
                    color="#222222",
                    s=16,
                    zorder=3,
                )
    axis.set(
        title="Mean final validation accuracy before vs after dropout path ban",
        xlabel="Grading mode × starter",
        ylabel="Mean final validation accuracy (%)",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, f"{note} · dots are per-seed finals", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, "003-compare-final-validation-by-score-metric.png"))

    # Mean dropout actions before vs after.
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    for phase_index, (phase_runs, phase_key) in enumerate(
        ((before, PHASE_BEFORE), (after, PHASE_AFTER))
    ):
        means = []
        for score_metric in SCORE_METRICS:
            for model in MODELS:
                group = [
                    run
                    for run in phase_runs
                    if run["status"] == "completed"
                    and run["score_metric"] == score_metric
                    and run["model"] == model
                ]
                means.append(_mean([float(run["dropout_actions"]) for run in group]))
        xs = [position + (phase_index - 0.5) * width for position in positions]
        bars = axis.bar(
            xs,
            means,
            width=width,
            color=PHASE_COLORS[phase_key],
            label=PHASE_LABELS[phase_key],
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        for position, (score_metric, model) in enumerate(
            (score_metric, model)
            for score_metric in SCORE_METRICS
            for model in MODELS
        ):
            for run in phase_runs:
                if (
                    run["status"] != "completed"
                    or run["score_metric"] != score_metric
                    or run["model"] != model
                ):
                    continue
                axis.scatter(
                    position + (phase_index - 0.5) * width,
                    float(run["dropout_actions"]),
                    color="#222222",
                    s=16,
                    zorder=3,
                )
    axis.set(
        title="Mean sequential-dropout actions before vs after path ban",
        xlabel="Grading mode × starter",
        ylabel="Mean dropout actions per completed seed",
        xticks=positions,
        xticklabels=labels,
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, note, ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, "003-compare-dropout-actions-by-score-metric.png"))

    # Clear answer: before vs after overall (all seeds pooled).
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    phase_means = [
        _mean([float(run["final_acc"]) * 100 for run in before]),
        _mean([float(run["final_acc"]) * 100 for run in after]),
    ]
    phase_keys = [PHASE_BEFORE, PHASE_AFTER]
    bars = axis.bar(
        [PHASE_LABELS[key] for key in phase_keys],
        phase_means,
        color=[PHASE_COLORS[key] for key in phase_keys],
        width=0.55,
    )
    axis.bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
    for phase_index, phase_runs in enumerate((before, after)):
        for run in phase_runs:
            if run["status"] != "completed":
                continue
            axis.scatter(
                phase_index,
                float(run["final_acc"]) * 100,
                color="#222222",
                s=18,
                zorder=3,
            )
    better_phase = PHASE_AFTER if phase_means[1] >= phase_means[0] else PHASE_BEFORE
    axis.set(
        title=f"Overall mean final validation: {PHASE_LABELS[better_phase]} is better",
        xlabel="Experiment phase",
        ylabel="Mean final validation accuracy (%)",
        ylim=(0, 100),
    )
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · pools all grading modes and starters · dots are seeds",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, "003-compare-overall-before-after.png"))

    # Clear answer: grade val vs grade train, shown for both phases.
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    width = 0.36
    phase_positions = [0, 1]
    for metric_index, score_metric in enumerate(SCORE_METRICS):
        means = []
        for phase_runs in (before, after):
            group = [
                run
                for run in phase_runs
                if run["status"] == "completed" and run["score_metric"] == score_metric
            ]
            means.append(_mean([float(run["final_acc"]) * 100 for run in group]))
        xs = [position + (metric_index - 0.5) * width for position in phase_positions]
        bars = axis.bar(
            xs,
            means,
            width=width,
            color=SCORE_COLORS[score_metric],
            label=_short(score_metric),
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        for phase_index, phase_runs in enumerate((before, after)):
            for run in phase_runs:
                if run["status"] != "completed" or run["score_metric"] != score_metric:
                    continue
                axis.scatter(
                    phase_index + (metric_index - 0.5) * width,
                    float(run["final_acc"]) * 100,
                    color="#222222",
                    s=16,
                    zorder=3,
                )
    axis.set(
        title="Which grading wins: grade val vs grade train by phase",
        xlabel="Experiment phase",
        ylabel="Mean final validation accuracy (%)",
        xticks=phase_positions,
        xticklabels=[PHASE_LABELS[PHASE_BEFORE], PHASE_LABELS[PHASE_AFTER]],
        ylim=(0, 100),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · each bar pools both starters · dots are seeds",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(save(figure, "003-compare-grading-overall-by-phase.png"))

    return written


def generate_charts(
    before_runs_dir: Path = DEFAULT_BEFORE_RUNS,
    after_runs_dir: Path = DEFAULT_AFTER_RUNS,
    output_dir: Path = DEFAULT_OUTPUT,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
    runs_dir: Path | None = None,
) -> list[Path]:
    """
    Load before/after runs (or snapshot), refresh snapshot, write phase and compare charts.

    ``runs_dir`` remains as a single-folder override for older callers/tests: it is treated
    as the before_fix grid and after_fix is left empty.
    """
    if runs_dir is not None:
        before_runs_dir = runs_dir
        after_runs_dir = runs_dir / "__missing_after_fix__"

    before_raw = load_runs(before_runs_dir)
    after_raw = load_runs(after_runs_dir)
    if before_raw or after_raw:
        phases = {PHASE_BEFORE: before_raw, PHASE_AFTER: after_raw}
        write_snapshot(
            phases,
            snapshot_path,
            folders={
                PHASE_BEFORE: before_runs_dir.name,
                PHASE_AFTER: after_runs_dir.name,
            },
        )
    else:
        phases = load_phases_or_snapshot(before_runs_dir, after_runs_dir, snapshot_path)

    if not phases[PHASE_BEFORE] and not phases[PHASE_AFTER]:
        raise FileNotFoundError(
            f"No Exp 003 runs or snapshot found under {before_runs_dir} / {after_runs_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    written: list[Path] = []

    def save(figure: plt.Figure, name: str) -> Path:
        path = output_dir / name
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
        return path

    for phase in (PHASE_BEFORE, PHASE_AFTER):
        runs = phases[phase]
        completed = [run for run in runs if run["status"] == "completed"]
        if not completed:
            continue
        note = (
            f"Source: {len(completed)}/{len(runs)} completed Exp 003 "
            f"{PHASE_LABELS[phase]} runs"
        )
        _write_phase_charts(
            completed,
            output_dir,
            PHASE_PREFIXES[phase],
            note,
            save,
        )

    before_completed = [run for run in phases[PHASE_BEFORE] if run["status"] == "completed"]
    after_completed = [run for run in phases[PHASE_AFTER] if run["status"] == "completed"]
    if before_completed and after_completed:
        note = (
            f"Source: before {len(before_completed)} completed · "
            f"after {len(after_completed)} completed"
        )
        _write_compare_charts(before_completed, after_completed, note, save)

    # Keep unprefixed aliases pointing at before_fix charts for older links.
    for stem in PHASE_CHART_STEMS:
        before_path = output_dir / f"{PHASE_PREFIXES[PHASE_BEFORE]}-{stem}.png"
        alias_path = output_dir / f"003-{stem}.png"
        if before_path.exists():
            alias_path.write_bytes(before_path.read_bytes())
            if alias_path not in written:
                written.append(alias_path)

    return written


if __name__ == "__main__":
    paths = generate_charts()
    print(f"Wrote {len(paths)} chart(s)")
    for path in paths:
        print(path)
