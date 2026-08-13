"""Generate charts for Experiment 002: initial architectures."""

from __future__ import annotations

import json
from collections import defaultdict
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
    / "exp002_initial_architectures_after_fix_1"
)
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-002-initial-architectures.json"

# Topology-only after_fix_1 starters, largest start params first.
MODELS = (
    "big",
    "medium_1conv_2linear",
    "medium_2conv_1linear",
    "small",
)
MODEL_COLORS = {
    "big": "#3568a8",
    "medium_1conv_2linear": "#4f8a63",
    "medium_2conv_1linear": "#2a8a8a",
    "small": "#d18b2c",
}
SEED_COLORS = {100: "#3568a8", 101: "#4f8a63", 102: "#d18b2c", 103: "#7a5a9a"}
ORDER_LABELS = ("1st", "2nd", "3rd", "4th", "5th+")
GENERATIONS = 5
# Corrected grid has actions only in generations 0–3; generation 4 stayed empty.
ACTION_CHART_GENERATIONS = (0, 1, 2, 3)
SHORT_MODEL_NAMES = {
    "medium_1conv_2linear": "med 1c+2l",
    "medium_2conv_1linear": "med 2c+1l",
}
# Collapsed big seeds that start with stacked dropout and never learn.
BIG_OUTLIER_SEEDS = frozenset({100, 101})


def _is_big_outlier(run: dict[str, object]) -> bool:
    return str(run["model"]) == "big" and int(run["seed"]) in BIG_OUTLIER_SEEDS


def _without_big_outliers(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [run for run in runs if not _is_big_outlier(run)]


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics for every architecture × seed run."""
    runs: list[dict[str, object]] = []
    if not runs_dir.exists():
        return runs
    for main_path in sorted(runs_dir.rglob("board/main.json")):
        run_dir = main_path.parent.parent
        parts = run_dir.relative_to(runs_dir).parts
        metrics_path = main_path.parent / "metrics" / "training.json"
        if not metrics_path.exists():
            continue
        main = json.loads(main_path.read_text(encoding="utf-8"))
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        actions = [
            (item["generation"], item["actionExecuted"])
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        runs.append(
            {
                "model": parts[0],
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "elapsed_sec": main.get("trainingTimeElapsedSec"),
                "started_on": main.get("experimentStartedOn"),
                "last_update": main.get("lastUpdate"),
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


def load_runs_or_snapshot(runs_dir: Path, snapshot_path: Path) -> list[dict[str, object]]:
    """Prefer raw runs; fall back to the committed snapshot."""
    runs = load_runs(runs_dir)
    if runs:
        return runs
    if not snapshot_path.exists():
        return []
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return list(payload.get("runs", []))


def write_snapshot(runs: list[dict[str, object]], snapshot_path: Path) -> None:
    """Persist a compact JSON snapshot for documentation without raw output."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    compact = []
    for run in runs:
        compact.append(
            {
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
                "action_epochs": run["action_epochs"],
                "action_labels": run["action_labels"],
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
        )
    snapshot_path.write_text(json.dumps({"runs": compact}, indent=2), encoding="utf-8")


# Completed after_fix_1 runs are the published analysis set.
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _completed(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [run for run in runs if run["status"] == "completed"]


def _analysis_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Completed runs used for ranking and action charts."""
    return _completed(runs)


def _models_by_start_params(
    runs: list[dict[str, object]],
    *,
    descending: bool = True,
) -> list[str]:
    """Order present architectures by mean starting parameter count."""
    present = {str(run["model"]) for run in runs}
    start_by_model: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        start_by_model[str(run["model"])].append(float(run["start_params"]))
    model_rank = {model: index for index, model in enumerate(MODELS)}
    candidates = [model for model in present]
    # Largest start params first; ties keep the script MODEL_VARIANTS order.
    return sorted(
        candidates,
        key=lambda model: (
            -_mean(start_by_model[model]) if descending else _mean(start_by_model[model]),
            model_rank.get(model, len(MODELS)),
            model,
        ),
    )


def _generations(run: dict[str, object]) -> dict[int, list[dict[str, object]]]:
    generations: dict[int, list[dict[str, object]]] = defaultdict(list)
    for epoch in list(run["epochs"]):
        if "generation" in epoch and epoch["generation"] is not None:
            generations[int(epoch["generation"])].append(epoch)
        else:
            generations[int(epoch["globalEpoch"]) // 10].append(epoch)
    return generations


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


def _plot_order_pair_bars(
    axis: Axes,
    train_order_values: list[list[float]],
    val_order_values: list[list[float]],
    title: str,
) -> None:
    positions = list(range(len(ORDER_LABELS)))
    train_means = [_mean(values) * 100 if values else 0.0 for values in train_order_values]
    val_means = [_mean(values) * 100 if values else 0.0 for values in val_order_values]
    axis.bar(
        [position - 0.18 for position in positions],
        train_means,
        width=0.36,
        color="#3568a8",
        alpha=0.45,
        label="Training",
    )
    axis.bar(
        [position + 0.18 for position in positions],
        val_means,
        width=0.36,
        color="#4f8a63",
        alpha=0.45,
        label="Validation",
    )
    for category, (train_values, val_values) in enumerate(
        zip(train_order_values, val_order_values, strict=True)
    ):
        for values, center, color in (
            (train_values, category - 0.18, "#3568a8"),
            (val_values, category + 0.18, "#4f8a63"),
        ):
            count = len(values)
            if count == 0:
                continue
            offsets = (
                [0.0]
                if count == 1
                else [-0.12 + 0.24 * index / (count - 1) for index in range(count)]
            )
            axis.scatter(
                [center + offset for offset in offsets],
                [value * 100 for value in values],
                color=color,
                s=14,
                alpha=0.7,
                edgecolor="#222222",
                linewidth=0.3,
            )
    axis.axhline(0, color="#222222", linewidth=1)
    axis.set_title(title)
    axis.set_xticks(positions)
    axis.set_xticklabels(ORDER_LABELS)
    axis.set_xlabel("Order of the action in one run")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)


def _short_action_name(name: str) -> str:
    return name.replace(" Action", "").replace("Add ", "+").replace("Delete ", "−")


def _short_model(name: str) -> str:
    return SHORT_MODEL_NAMES.get(name, name)


def generate_charts(
    runs_dir: Path = DEFAULT_RUNS,
    output_dir: Path = DEFAULT_OUTPUT,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
) -> list[Path]:
    """Load runs (or snapshot), refresh snapshot when raw exists, write charts."""
    raw_runs = load_runs(runs_dir)
    if raw_runs:
        write_snapshot(raw_runs, snapshot_path)
        runs = raw_runs
    else:
        runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    if not runs:
        raise FileNotFoundError(f"No experiment runs or snapshot found for {runs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    completed = _analysis_runs(runs)
    models = _models_by_start_params(completed) if completed else _models_by_start_params(runs)
    n_complete = len(completed)
    n_loaded_complete = len(_completed(runs))
    n_total = len(runs)
    note = f"Source: {n_complete} completed after_fix_1 runs ({n_loaded_complete}/{n_total} loaded)"
    order_note = "largest starting parameters first"
    written: list[Path] = []

    def save(figure: plt.Figure, name: str) -> Path:
        path = output_dir / name
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
        return path

    positions = list(range(len(models)))
    filtered = _without_big_outliers(completed)
    filtered_note = (
        f"Source: {len(filtered)} runs after removing big seeds "
        f"{sorted(BIG_OUTLIER_SEEDS)} (early stacked dropout collapses)"
    )

    def plot_final_accuracy(
        runs_for_plot: list[dict[str, object]],
        filename: str,
        title: str,
        footer: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(11.5, 5.2))
        for metric, offset, color, label in (
            ("final_train_acc", -0.18, "#3568a8", "Training accuracy"),
            ("final_acc", 0.18, "#4f8a63", "Validation accuracy"),
        ):
            means = [
                _mean([float(run[metric]) * 100 for run in runs_for_plot if run["model"] == model])
                for model in models
            ]
            bars = axis.bar(
                [position + offset for position in positions],
                means,
                width=0.36,
                color=color,
                label=label,
            )
            axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        for model_index, model in enumerate(models):
            for run in runs_for_plot:
                if run["model"] != model:
                    continue
                axis.scatter(
                    model_index - 0.18,
                    float(run["final_train_acc"]) * 100,
                    color="#1f3f6d",
                    s=18,
                    zorder=3,
                )
                axis.scatter(
                    model_index + 0.18,
                    float(run["final_acc"]) * 100,
                    color="#222222",
                    s=18,
                    zorder=3,
                )
        axis.set(
            title=title,
            xlabel=f"Initial architecture ({order_note})",
            ylabel="Mean final accuracy (%)",
            xticks=positions,
            xticklabels=[_short_model(model) for model in models],
            ylim=(0, 100),
        )
        axis.tick_params(axis="x", rotation=25)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.text(0.99, 0.01, footer, ha="right", fontsize=7)
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        save(figure, filename)

    def plot_param_growth(
        runs_for_plot: list[dict[str, object]],
        filename: str,
        title: str,
        footer: str,
    ) -> None:
        figure, axis = plt.subplots(figsize=(11.5, 4.8))
        for model_index, model in enumerate(models):
            group = [run for run in runs_for_plot if run["model"] == model]
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
                color=MODEL_COLORS.get(model, "#3568a8"),
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
            title=title,
            xlabel=f"Initial architecture ({order_note})",
            ylabel="Parameter count",
            xticks=positions,
            xticklabels=[_short_model(model) for model in models],
        )
        axis.tick_params(axis="x", rotation=25)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.text(0.99, 0.01, footer, ha="right", fontsize=7)
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        save(figure, filename)

    plot_final_accuracy(
        completed,
        "002-final-accuracy-by-architecture.png",
        "Mean final training and validation accuracy by initial architecture",
        f"{note} · {order_note} · dots are per-seed finals (blue=train, black=val)",
    )
    plot_final_accuracy(
        filtered,
        "002-final-accuracy-without-big-outliers.png",
        "Mean final accuracy without collapsed big seeds 100 and 101",
        f"{filtered_note} · {order_note} · dots are per-seed finals (blue=train, black=val)",
    )

    # Best-seed envelope: avoids averaging in collapsed outlier seeds.
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    for metric, offset, color, label in (
        ("final_train_acc", -0.18, "#3568a8", "Best-seed final training"),
        ("final_acc", 0.18, "#4f8a63", "Best-seed final validation"),
    ):
        peaks = []
        for model in models:
            values = [float(run[metric]) * 100 for run in completed if run["model"] == model]
            peaks.append(max(values) if values else 0.0)
        bars = axis.bar(
            [position + offset for position in positions],
            peaks,
            width=0.36,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    axis.set(
        title="Best-seed final training and validation accuracy by initial architecture",
        xlabel=f"Initial architecture ({order_note})",
        ylabel="Best final accuracy among completed seeds (%)",
        xticks=positions,
        xticklabels=[_short_model(model) for model in models],
        ylim=(0, 100),
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · {order_note} · each bar is the best seed, not the mean",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-best-seed-accuracy-by-architecture.png")

    plot_param_growth(
        completed,
        "002-param-growth.png",
        "Starting and final parameter counts",
        f"{note} · {order_note} · dots show final parameters per completed seed",
    )
    plot_param_growth(
        filtered,
        "002-param-growth-without-big-outliers.png",
        "Parameter growth without collapsed big seeds 100 and 101",
        f"{filtered_note} · {order_note} · dots show final parameters per remaining seed",
    )

    # Action counts by generation (0–3). Generation 4 had no actions in this grid.
    figure, axis = plt.subplots(figsize=(11.5, 5.0))
    width = min(0.8 / max(len(models), 1), 0.18)
    generation_positions = list(ACTION_CHART_GENERATIONS)
    for model_index, model in enumerate(models):
        group = [run for run in completed if run["model"] == model]
        totals = []
        for generation in ACTION_CHART_GENERATIONS:
            totals.append(
                sum(
                    1
                    for run in group
                    for action_generation in list(run["action_generations"])
                    if int(action_generation) == generation
                )
            )
        offset = (model_index - (len(models) - 1) / 2) * width
        bars = axis.bar(
            [position + offset for position in generation_positions],
            totals,
            width=width,
            color=MODEL_COLORS.get(model, "#3568a8"),
            label=_short_model(model),
        )
        axis.bar_label(bars, fmt="%d", fontsize=7, padding=1)
    axis.set(
        title="Executed action counts by generation and architecture",
        xlabel="Generation",
        ylabel="Action count across all seeds",
        xticks=generation_positions,
        xticklabels=[f"Gen {generation}" for generation in ACTION_CHART_GENERATIONS],
    )
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.text(
        0.99,
        0.01,
        f"{note} · totals across four seeds · generation 4 had zero actions",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-actions-by-generation.png")

    # Recovery-window gains for training and validation.
    train_gains_by_order: dict[int, list[float]] = defaultdict(list)
    val_gains_by_order: dict[int, list[float]] = defaultdict(list)
    train_gains_by_model_order: dict[str, dict[int, list[float]]] = {
        model: defaultdict(list) for model in models
    }
    val_gains_by_model_order: dict[str, dict[int, list[float]]] = {
        model: defaultdict(list) for model in models
    }
    effects_by_type: dict[str, list[float]] = defaultdict(list)
    train_effects_by_type: dict[str, list[float]] = defaultdict(list)

    for run in completed:
        generations = _generations(run)
        if not generations:
            continue
        action_map = dict(
            zip(list(run["action_generations"]), list(run["action_labels"]), strict=True)
        )
        model = str(run["model"])
        action_order = 0
        for generation in range(1, max(generations) + 1):
            if generation - 1 not in action_map:
                continue
            previous = generations[generation - 1]
            current = generations[generation]
            val_gain = float(current[-1]["valAcc"]) - float(previous[-1]["valAcc"])
            train_gain = float(current[-1]["trainAcc"]) - float(previous[-1]["trainAcc"])
            label = str(action_map[generation - 1])
            train_gains_by_order[action_order].append(train_gain)
            val_gains_by_order[action_order].append(val_gain)
            if model in train_gains_by_model_order:
                train_gains_by_model_order[model][action_order].append(train_gain)
                val_gains_by_model_order[model][action_order].append(val_gain)
            effects_by_type[label].append(val_gain)
            train_effects_by_type[label].append(train_gain)
            action_order += 1

    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    _plot_order_pair_bars(
        axis,
        _order_buckets(train_gains_by_order),
        _order_buckets(val_gains_by_order),
        "Training and validation change by action order",
    )
    axis.set_ylabel("Accuracy change over the next generation (percentage points)")
    figure.text(
        0.99,
        0.01,
        f"{note} · bars are means; dots are individual observed actions",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-action-order.png")

    panel_models = [
        model
        for model in models
        if any(val_gains_by_model_order[model].values())
        or any(train_gains_by_model_order[model].values())
    ]
    if panel_models:
        cols = min(2, len(panel_models))
        rows = (len(panel_models) + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows), sharey=True)
        flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for axis, model in zip(flat, panel_models, strict=False):
            _plot_order_pair_bars(
                axis,
                _order_buckets(train_gains_by_model_order[model]),
                _order_buckets(val_gains_by_model_order[model]),
                _short_model(model),
            )
        for axis in flat[len(panel_models) :]:
            axis.axis("off")
        flat[0].set_ylabel("Accuracy change over the next generation (percentage points)")
        figure.suptitle("Training and validation change by action order and architecture")
        figure.text(
            0.99,
            0.01,
            f"{note} · same recovery window · one panel per architecture",
            ha="right",
            fontsize=7,
        )
        figure.tight_layout(rect=(0, 0.03, 1, 0.94))
        save(figure, "002-action-order-by-architecture.png")

    type_names = sorted(effects_by_type)
    if type_names:
        short_names = [_short_action_name(name) for name in type_names]
        figure, axis = plt.subplots(figsize=(9.5, 4.8))
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
            f"{note} · bars are means; colored dots are individual observed actions",
            ha="right",
            fontsize=7,
        )
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        save(figure, "002-action-types.png")

    composition = {model: defaultdict(int) for model in models}
    for run in completed:
        for label in list(run["action_labels"]):
            composition[str(run["model"])][str(label)] += 1
    type_list = sorted({name for model in models for name in composition[model]})
    if type_list:
        figure, axis = plt.subplots(figsize=(11.5, 5.0))
        width = min(0.8 / max(len(models), 1), 0.14)
        for model_index, model in enumerate(models):
            counts = [composition[model][name] for name in type_list]
            offset = (model_index - (len(models) - 1) / 2) * width
            axis.bar(
                [index + offset for index in range(len(type_list))],
                counts,
                width=width,
                color=MODEL_COLORS.get(model, "#3568a8"),
                label=_short_model(model),
            )
        axis.set(
            title="Executed action counts by type and initial architecture",
            xlabel="Action type",
            ylabel="Number of executed actions across completed seeds",
            xticks=list(range(len(type_list))),
            xticklabels=[_short_action_name(name) for name in type_list],
        )
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=7, ncol=2)
        axis.grid(axis="y", alpha=0.25)
        figure.text(
            0.99,
            0.01,
            f"{note} · counts show what each starter actually searched",
            ha="right",
            fontsize=7,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        save(figure, "002-action-composition.png")

    # Training curves by architecture.
    if models:
        cols = min(3, len(models))
        rows = (len(models) + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), sharey=True)
        flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for axis, model in zip(flat, models, strict=False):
            for run in completed:
                if run["model"] != model:
                    continue
                epochs = list(run["epochs"])
                axis.plot(
                    [int(epoch["globalEpoch"]) for epoch in epochs],
                    [float(epoch["trainAcc"]) * 100 for epoch in epochs],
                    color=SEED_COLORS.get(int(run["seed"]), "#3568a8"),
                    alpha=0.85,
                    label=f"seed {run['seed']}",
                )
            axis.set_title(_short_model(model))
            axis.set_xlabel("Global epoch")
            axis.grid(alpha=0.2)
            handles, legend_labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, legend_labels, fontsize=6)
        for axis in flat[len(models) :]:
            axis.axis("off")
        flat[0].set_ylabel("Training accuracy (%)")
        figure.suptitle("Training-accuracy curves by initial architecture")
        figure.text(0.99, 0.01, f"{note} · board/metrics/training.json", ha="right", fontsize=7)
        figure.tight_layout(rect=(0, 0.03, 1, 0.94))
        save(figure, "002-training-curves.png")

    return written


if __name__ == "__main__":
    paths = generate_charts()
    print(f"Wrote {len(paths)} chart(s)")
    for path in paths:
        print(path)
