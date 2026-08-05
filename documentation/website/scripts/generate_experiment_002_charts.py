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
    / "exp002_initial_architectures"
)
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-002-initial-architectures.json"

# Prefer known names for stable ordering; unknown models fall back to start params.
MODELS = (
    "big",
    "medium",
    "medium_avg_pool_only",
    "medium_max_pool_only",
    "small_avg_pool_only",
    "small_max_pool_only",
    "big_ch2_h8",
    "medium_ch2_h8",
    "medium_h4",
    "very_small",
    "very_small_ch2",
    "very_small_avg_pool_only",
    "very_small_max_pool_only",
)
MODEL_COLORS = {
    "big": "#3568a8",
    "medium": "#4f8a63",
    "medium_avg_pool_only": "#708050",
    "medium_max_pool_only": "#5a7a40",
    "small_avg_pool_only": "#3a6a8a",
    "small_max_pool_only": "#2a5a7a",
    "big_ch2_h8": "#8a4a3a",
    "medium_ch2_h8": "#2a8a8a",
    "medium_h4": "#7a5a9a",
    "very_small": "#d18b2c",
    "very_small_ch2": "#a07030",
    "very_small_avg_pool_only": "#c08020",
    "very_small_max_pool_only": "#906018",
}
SEED_COLORS = {100: "#3568a8", 101: "#4f8a63", 102: "#d18b2c", 103: "#7a5a9a"}
ORDER_LABELS = ("1st", "2nd", "3rd", "4th", "5th+")
PHASES = (("Early 0–3", 0, 3), ("Middle 4–6", 4, 6), ("Late 7–9", 7, 9))


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


# Old flatten controls start far above compact stems (~12k / ~50k params).
# The revised Exp 002 `medium_max_pool_only` is compact (~276 params) and stays in charts.
OVERSIZED_START_PARAMS = 1000


def _is_oversized_flatten_control(run: dict[str, object]) -> bool:
    """True for starters that flatten a large spatial map into the first linear."""
    model = str(run["model"])
    start_params = int(run["start_params"])
    if model == "medium_no_pool":
        return True
    return model == "medium_max_pool_only" and start_params > OVERSIZED_START_PARAMS


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _completed(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [run for run in runs if run["status"] == "completed"]


def _analysis_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Completed runs used for ranking and action charts."""
    return [
        run
        for run in _completed(runs)
        if not _is_oversized_flatten_control(run)
    ]


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


def _short_model(name: str) -> str:
    return name.replace("medium_", "m_").replace("very_small", "vsmall")


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
    note = (
        f"Source: {n_complete} compact completed runs"
        f" ({n_loaded_complete}/{n_total} loaded; "
        "oversized flatten controls ignored)"
    )
    order_note = "largest starting parameters first"
    written: list[Path] = []

    def save(figure: plt.Figure, name: str) -> Path:
        path = output_dir / name
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
        return path

    positions = list(range(len(models)))

    # Final train/val accuracy by architecture (ordered by start params).
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    for metric, offset, color, label in (
        ("final_train_acc", -0.18, "#3568a8", "Training accuracy"),
        ("final_acc", 0.18, "#4f8a63", "Validation accuracy"),
    ):
        means = [
            _mean([float(run[metric]) * 100 for run in completed if run["model"] == model])
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
        for run in completed:
            if run["model"] != model:
                continue
            axis.scatter(
                model_index + 0.18,
                float(run["final_acc"]) * 100,
                color="#222222",
                s=18,
                zorder=3,
            )
    axis.set(
        title="Mean final training and validation accuracy by initial architecture",
        xlabel=f"Initial architecture ({order_note})",
        ylabel="Mean final accuracy across completed seeds (%)",
        xticks=positions,
        xticklabels=[_short_model(model) for model in models],
        ylim=(0, 100),
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(0.99, 0.01, f"{note} · {order_note} · dots are final validation per seed", ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-final-accuracy-by-architecture.png")

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

    # Mean end-of-generation training accuracy: find when runs near the strong band.
    figure, axis = plt.subplots(figsize=(12.0, 5.2))
    width = min(0.8 / max(len(models), 1), 0.12)
    strong_band = 91.0
    for model_index, model in enumerate(models):
        group = [run for run in completed if run["model"] == model]
        means = []
        for generation in range(10):
            values = []
            for run in group:
                generations = _generations(run)
                if generation not in generations:
                    continue
                values.append(float(generations[generation][-1]["trainAcc"]) * 100)
            means.append(_mean(values))
        offset = (model_index - (len(models) - 1) / 2) * width
        axis.bar(
            [generation + offset for generation in range(10)],
            means,
            width=width,
            color=MODEL_COLORS.get(model, "#3568a8"),
            label=_short_model(model),
        )
    axis.axhline(
        strong_band,
        color="#a65353",
        linestyle="--",
        linewidth=1.4,
        label=f"{strong_band:.0f}% strong band",
    )
    axis.set(
        title="Mean end-of-generation training accuracy by architecture",
        xlabel="Generation",
        ylabel="Mean training accuracy (%)",
        xticks=list(range(10)),
        ylim=(0, 100),
    )
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · dashed line marks {strong_band:.0f}% training accuracy",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-train-acc-by-generation.png")

    # Parameter growth ordered by starting size.
    figure, axis = plt.subplots(figsize=(11.5, 4.8))
    for model_index, model in enumerate(models):
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
        title="Starting and final parameter counts",
        xlabel=f"Initial architecture ({order_note})",
        ylabel="Parameter count",
        xticks=positions,
        xticklabels=[_short_model(model) for model in models],
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.99,
        0.01,
        f"{note} · {order_note} · dots show final parameters per completed seed",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-param-growth.png")

    # Actions by early / middle / late generation phases.
    figure, axis = plt.subplots(figsize=(11.5, 5.0))
    width = min(0.8 / max(len(models), 1), 0.18)
    phase_positions = list(range(len(PHASES)))
    for model_index, model in enumerate(models):
        group = [run for run in completed if run["model"] == model]
        means = []
        for _, lo, hi in PHASES:
            counts = [
                sum(1 for generation in list(run["action_generations"]) if lo <= int(generation) <= hi)
                for run in group
            ]
            means.append(_mean([float(value) for value in counts]))
        offset = (model_index - (len(models) - 1) / 2) * width
        axis.bar(
            [position + offset for position in phase_positions],
            means,
            width=width,
            color=MODEL_COLORS.get(model, "#3568a8"),
            label=_short_model(model),
        )
    axis.set(
        title="Mean executed actions by training phase and architecture",
        xlabel="Generation phase",
        ylabel="Mean actions per completed seed",
        xticks=phase_positions,
        xticklabels=[label for label, _, _ in PHASES],
    )
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.text(
        0.99,
        0.01,
        f"{note} · early = gens 0–3, middle = gens 4–6, late = gens 7–9",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-actions-by-phase.png")

    # Recovery-window gains.
    action_gains_by_order: dict[int, list[float]] = defaultdict(list)
    action_gains_by_model_order: dict[str, dict[int, list[float]]] = {
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
            action_gains_by_order[action_order].append(val_gain)
            if model in action_gains_by_model_order:
                action_gains_by_model_order[model][action_order].append(val_gain)
            effects_by_type[label].append(val_gain)
            train_effects_by_type[label].append(train_gain)
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
        f"{note} · bars are means; dots are individual observed actions",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "002-action-order.png")

    panel_models = [model for model in models if any(action_gains_by_model_order[model].values())]
    if panel_models:
        cols = min(3, len(panel_models))
        rows = (len(panel_models) + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), sharey=True)
        flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for axis, model in zip(flat, panel_models, strict=False):
            _plot_order_bars(
                axis,
                _order_buckets(action_gains_by_model_order[model]),
                MODEL_COLORS.get(model, "#3568a8"),
                _short_model(model),
            )
        for axis in flat[len(panel_models) :]:
            axis.axis("off")
        flat[0].set_ylabel(
            "Validation-accuracy change over the next generation (percentage points)"
        )
        figure.suptitle("Validation-accuracy change by action order and architecture")
        figure.text(
            0.99,
            0.01,
            f"{note} · same recovery window · one panel per architecture with completed seeds",
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
