"""Generate charts for Experiment 004: composed vs recovery-only LR schedules.

Measured figures (boards or snapshot):
- small explainer strips for each schedule shape
- per-schedule LR for seeds 100/101 with base, effective, and recovery factor
- per-schedule training and validation accuracy for seeds 100/101
- post-action training-accuracy change by schedule
- final accuracy bars
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.train_mnist_exp004_composed_lr_schedulers import (
    build_learning_rate_scheduler_for_schedule_id,
)
from growingnn.training.lr_scheduler_action import compute_schedule_value_without_advancing
from growingnn.training.lr_scheduler_global import ComposedLearningRateScheduler

SITE = Path(__file__).parents[1]
_RUNS_ROOT = SITE.parents[1] / "experiments" / "output" / "train_mnist" / "runs"
DEFAULT_RUNS = _RUNS_ROOT / "exp004_composed_lr_schedulers"
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-004-composed-lr-schedulers.json"
_ALLOWED_SNAPSHOT_ROOT = (SITE / "data" / "experiments").resolve()
_ALLOWED_OUTPUT_ROOT = (SITE / "app" / "public" / "assets" / "experiments").resolve()
_ALLOWED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


def _resolve_under_allowed_root(path: Path, allowed_root: Path) -> Path:
    """Resolve *path* and reject reads/writes outside *allowed_root* or the system temp dir."""
    if ".." in path.parts:
        raise ValueError("path must not contain '..'")
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(allowed_root) or resolved.is_relative_to(_ALLOWED_TEMP_ROOT):
        return resolved
    raise ValueError(
        f"path {resolved} is outside allowed roots {allowed_root} and {_ALLOWED_TEMP_ROOT}"
    )


SCHEDULE_IDS = (
    "recovery_only_logistic",
    "composed_cosine",
    "composed_step",
    "composed_exponential",
    "composed_linear",
    "composed_constant",
    "composed_linear_1_to_0p1",
)
SCHEDULE_LABELS = {
    "recovery_only_logistic": "recovery-only logistic",
    "composed_cosine": "composed cosine",
    "composed_step": "composed step",
    "composed_exponential": "composed exponential",
    "composed_linear": "composed linear",
    "composed_constant": "composed constant",
    "composed_linear_1_to_0p1": "cascade 1.0→0.1",
}
SCHEDULE_BLURBS = {
    "recovery_only_logistic": "Exp 003 style. Absolute logistic warmup only. No global decay.",
    "composed_cosine": "Cosine base from 0.01 to 0.001, times logistic recovery.",
    "composed_step": "Step base drops by 0.5 every 33 epochs, times logistic recovery.",
    "composed_exponential": "Exponential base with gamma 0.98, times logistic recovery.",
    "composed_linear": "Linear base from 0.01 to 0.001, times logistic recovery.",
    "composed_constant": "Flat base 0.01, times logistic recovery after actions.",
    "composed_linear_1_to_0p1": "Custom cascade base from 1.0 to 0.1, times logistic recovery.",
}
SEED_COLORS = {100: "#3568a8", 101: "#4f8a63", 102: "#d18b2c"}
DISPLAY_SEEDS = (100, 101)
EPOCHS_PER_GENERATION = 10
TOTAL_EPOCHS = 100
COLOR_EFFECTIVE = "#3568a8"
COLOR_BASE = "#d18b2c"
COLOR_FACTOR = "#4f8a63"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _short(schedule_id: str) -> str:
    return SCHEDULE_LABELS.get(schedule_id, schedule_id)


def autoscaled_y_limits(values: list[float], pad_ratio: float = 0.08) -> tuple[float, float]:
    """Zoom y-limits to the series range so small LR bands stay readable."""
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if high <= low:
        pad = max(abs(high) * 0.05, 1e-4)
        return low - pad, high + pad
    span = high - low
    pad = max(span * pad_ratio, abs(high) * 0.02, 1e-6)
    return low - pad, high + pad


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics for every schedule × seed run."""
    runs: list[dict[str, object]] = []
    if not runs_dir.exists():
        return runs
    if ".." in runs_dir.parts:
        raise ValueError("path must not contain '..'")
    resolved_runs = runs_dir.expanduser().resolve()
    allowed_runs_root = _RUNS_ROOT.resolve()
    if not (
        resolved_runs.is_relative_to(allowed_runs_root)
        or resolved_runs.is_relative_to(_ALLOWED_TEMP_ROOT)
    ):
        raise ValueError(
            f"path {resolved_runs} is outside allowed roots "
            f"{allowed_runs_root} and {_ALLOWED_TEMP_ROOT}"
        )
    for main_path in sorted(resolved_runs.rglob("board/main.json")):
        main_resolved = main_path.resolve()
        if not main_resolved.is_relative_to(resolved_runs):
            continue
        run_dir = main_resolved.parent.parent
        parts = run_dir.relative_to(resolved_runs).parts
        metrics_path = main_resolved.parent / "metrics" / "training.json"
        if not metrics_path.exists() or len(parts) < 2:
            continue
        schedule_id = parts[0]
        main = json.loads(main_resolved.read_text(encoding="utf-8"))
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        actions = [
            (item["generation"], item["actionExecuted"])
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        runs.append(
            {
                "schedule_id": schedule_id,
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "elapsed_sec": main.get("trainingTimeElapsedSec"),
                "started_on": main.get("experimentStartedOn"),
                "last_update": main.get("lastUpdate"),
                "actions": len(actions),
                "action_generations": [generation for generation, _ in actions],
                "action_labels": [action["shortLabel"] for _, action in actions],
                "train_acc": [float(row["trainAcc"]) for row in epochs],
                "val_acc": [float(row["valAcc"]) for row in epochs],
                "lr": [float(row["lr"]) for row in epochs],
                "final_train_acc": float(epochs[-1]["trainAcc"]) if epochs else 0.0,
                "final_val_acc": float(epochs[-1]["valAcc"]) if epochs else 0.0,
            }
        )
    return runs


def write_snapshot(runs: list[dict[str, object]], snapshot_path: Path, folder: str) -> None:
    resolved = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "004", "folder": folder, "runs": runs}
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_runs_or_snapshot(runs_dir: Path, snapshot_path: Path) -> list[dict[str, object]]:
    runs = load_runs(runs_dir)
    if runs:
        write_snapshot(runs, snapshot_path, runs_dir.name)
        return runs
    resolved_snapshot = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    if not resolved_snapshot.exists():
        return []
    payload = json.loads(resolved_snapshot.read_text(encoding="utf-8"))
    return list(payload.get("runs", []))


def _scheduler_for_chart(schedule_id: str):
    return build_learning_rate_scheduler_for_schedule_id(
        schedule_id,
        {"lr_alpha": 0.01, "generations": 10, "epochs": 10},
    )


def simulate_lr_components(
    schedule_id: str,
    action_generations: list[int],
    n_epochs: int = TOTAL_EPOCHS,
) -> dict[str, list[float]]:
    """
    Replay base, recovery factor, and effective LR for one schedule and action list.

    An action after generation g resets recovery at the start of epoch (g+1)*10.
    """
    scheduler = _scheduler_for_chart(schedule_id)
    reset_epochs = {(int(generation) + 1) * EPOCHS_PER_GENERATION for generation in action_generations}
    bases: list[float] = []
    factors: list[float] = []
    effectives: list[float] = []

    for epoch in range(n_epochs):
        if epoch in reset_epochs:
            scheduler.structure_changed()

        if isinstance(scheduler, ComposedLearningRateScheduler):
            base = float(scheduler.global_schedule.lr_at(scheduler.global_epoch, scheduler.total_epochs))
            factor = float(
                compute_schedule_value_without_advancing(
                    scheduler.recovery._schedule,
                    0,
                    n_epochs,
                )
            )
            effective = float(scheduler.alpha_scheduler(epoch % EPOCHS_PER_GENERATION, EPOCHS_PER_GENERATION))
        else:
            # Absolute warmup: peak is fixed alpha; factor is value/alpha before floor effects.
            peak = float(scheduler._schedule.alpha)
            base = peak
            before = float(
                compute_schedule_value_without_advancing(scheduler._schedule, epoch, n_epochs)
            )
            factor = 0.0 if peak == 0 else max(0.0, min(1.0, before / peak))
            effective = float(scheduler.alpha_scheduler(epoch, n_epochs))

        bases.append(base)
        factors.append(factor)
        effectives.append(effective)

    return {"base": bases, "factor": factors, "effective": effectives}


def post_action_train_accuracy_changes(
    completed: list[dict[str, object]],
) -> dict[str, list[float]]:
    """
    Train-accuracy change over one recovery generation after each action.

    Action after generation g: compare last train acc of gen g with last train acc of gen g+1.
    Values are percentage points.
    """
    by_schedule: dict[str, list[float]] = defaultdict(list)
    for run in completed:
        train = [100.0 * float(value) for value in run["train_acc"]]
        for generation in run.get("action_generations", []):
            generation = int(generation)
            end_before = (generation + 1) * EPOCHS_PER_GENERATION - 1
            end_after = (generation + 2) * EPOCHS_PER_GENERATION - 1
            if end_after >= len(train) or end_before < 0:
                continue
            by_schedule[str(run["schedule_id"])].append(train[end_after] - train[end_before])
    return by_schedule


def _run_for_schedule_seed(
    completed: list[dict[str, object]],
    schedule_id: str,
    seed: int,
) -> dict[str, object] | None:
    for run in completed:
        if run["schedule_id"] == schedule_id and int(run["seed"]) == seed:
            return run
    return None


def _plot_explainer_strip() -> plt.Figure:
    """Small one-dip demos so readers can see each schedule shape before measured plots."""
    figure, axes = plt.subplots(1, len(SCHEDULE_IDS), figsize=(14, 2.6), sharey=False)
    for axis, schedule_id in zip(axes, SCHEDULE_IDS):
        components = simulate_lr_components(schedule_id, action_generations=[3], n_epochs=TOTAL_EPOCHS)
        axis.plot(components["effective"], color=COLOR_EFFECTIVE, linewidth=1.4)
        low, high = autoscaled_y_limits(components["effective"])
        axis.set_ylim(low, high)
        axis.set_title(_short(schedule_id), fontsize=7)
        axis.set_xticks([0, 50, 99])
        axis.tick_params(labelsize=6)
        axis.grid(True, alpha=0.2)
    figure.suptitle("Quick shape guide (one demo action after generation 3)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    return figure


def _plot_lr_components_for_schedule(
    completed: list[dict[str, object]],
    schedule_id: str,
) -> plt.Figure | None:
    """
    Seeds 100/101 side by side.

    Top row: base + measured effective LR.
    Bottom row: recovery factor.
    """
    figure = plt.figure(figsize=(11, 5.2))
    grid = GridSpec(2, 2, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.28)
    plotted = 0
    for column, seed in enumerate(DISPLAY_SEEDS):
        run = _run_for_schedule_seed(completed, schedule_id, seed)
        axis_lr = figure.add_subplot(grid[0, column])
        axis_factor = figure.add_subplot(grid[1, column], sharex=axis_lr)
        if run is None:
            axis_lr.set_title(f"{_short(schedule_id)} · seed {seed} (missing)")
            axis_lr.axis("off")
            axis_factor.axis("off")
            continue

        components = simulate_lr_components(
            schedule_id,
            list(run.get("action_generations", [])),
            n_epochs=len(run["lr"]),
        )
        measured = [float(value) for value in run["lr"]]
        axis_lr.plot(
            components["base"],
            color=COLOR_BASE,
            linewidth=1.4,
            linestyle="--",
            label="base LR",
        )
        axis_lr.plot(
            measured,
            color=COLOR_EFFECTIVE,
            linewidth=1.8,
            label="effective LR (measured)",
        )
        for generation in run.get("action_generations", []):
            epoch_index = (int(generation) + 1) * EPOCHS_PER_GENERATION
            if 0 <= epoch_index < len(measured):
                axis_lr.axvline(epoch_index, color="#999999", linestyle=":", linewidth=0.9, alpha=0.8)
                axis_factor.axvline(epoch_index, color="#999999", linestyle=":", linewidth=0.9, alpha=0.8)
        low, high = autoscaled_y_limits(measured + components["base"])
        axis_lr.set_ylim(low, high)
        axis_lr.set_title(f"{_short(schedule_id)} · seed {seed}")
        axis_lr.set_ylabel("learning rate")
        axis_lr.grid(True, alpha=0.25)
        axis_lr.legend(fontsize=7, loc="best")

        axis_factor.plot(components["factor"], color=COLOR_FACTOR, linewidth=1.6, label="recovery factor")
        axis_factor.set_ylim(-0.05, 1.05)
        axis_factor.set_xlabel("epoch index")
        axis_factor.set_ylabel("recovery factor")
        axis_factor.grid(True, alpha=0.25)
        axis_factor.legend(fontsize=7, loc="best")
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        return None
    figure.suptitle(
        f"LR parts for {_short(schedule_id)}: base, effective, recovery",
        fontsize=11,
    )
    return figure


def _plot_dual_seed_metric(
    completed: list[dict[str, object]],
    schedule_id: str,
    metric_key: str,
    *,
    y_label: str,
    title_prefix: str,
    scale: float = 1.0,
) -> plt.Figure | None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    plotted = 0
    for axis, seed in zip(axes, DISPLAY_SEEDS):
        run = _run_for_schedule_seed(completed, schedule_id, seed)
        if run is None:
            axis.set_title(f"{_short(schedule_id)} · seed {seed} (missing)")
            axis.axis("off")
            continue
        series = [scale * float(value) for value in run[metric_key]]
        axis.plot(
            range(len(series)),
            series,
            color=SEED_COLORS.get(seed, "#333333"),
            linewidth=1.8,
            label=f"seed {seed}",
        )
        for generation in run.get("action_generations", []):
            epoch_index = (int(generation) + 1) * EPOCHS_PER_GENERATION
            if 0 <= epoch_index < len(series):
                axis.axvline(epoch_index, color="#999999", linestyle="--", linewidth=0.9, alpha=0.7)
        low, high = autoscaled_y_limits(series)
        axis.set_ylim(low, high)
        axis.set_title(f"{_short(schedule_id)} · seed {seed}")
        axis.set_xlabel("epoch index")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, loc="best")
        plotted += 1
    if plotted == 0:
        plt.close(figure)
        return None
    figure.suptitle(f"{title_prefix}: {_short(schedule_id)}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def _plot_post_action_train_changes(
    by_schedule: dict[str, list[float]],
) -> plt.Figure | None:
    if not any(by_schedule.get(schedule_id) for schedule_id in SCHEDULE_IDS):
        return None
    figure, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharey=True)
    axes_flat = list(axes.flat)
    for axis, schedule_id in zip(axes_flat, SCHEDULE_IDS):
        values = by_schedule.get(schedule_id, [])
        if values:
            axis.axhline(0.0, color="#666666", linewidth=1.0)
            axis.scatter(
                range(len(values)),
                values,
                s=22,
                color="#3568a8",
                alpha=0.75,
                zorder=3,
            )
            axis.bar(
                [-0.6],
                [_mean(values)],
                width=0.5,
                color="#9bb7d4",
                label=f"mean {_mean(values):+.1f}",
            )
            negative = sum(1 for value in values if value < 0)
            axis.set_title(
                f"{_short(schedule_id)}\nneg {negative}/{len(values)}",
                fontsize=8,
            )
        else:
            axis.set_title(_short(schedule_id), fontsize=8)
        axis.set_xlabel("action index")
        axis.grid(True, axis="y", alpha=0.25)
    for axis in axes_flat[len(SCHEDULE_IDS) :]:
        axis.axis("off")
    axes_flat[0].set_ylabel("train acc change (percentage points)")
    figure.suptitle(
        "Train-accuracy change one generation after each architecture action",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def generate_charts(
    runs_dir: Path = DEFAULT_RUNS,
    output_dir: Path = DEFAULT_OUTPUT,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
) -> list[Path]:
    runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    completed = [run for run in runs if run.get("status") == "completed"]

    resolved_output = _resolve_under_allowed_root(output_dir, _ALLOWED_OUTPUT_ROOT)
    resolved_output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    written: list[Path] = []

    def save(figure: plt.Figure, name: str) -> None:
        path = resolved_output / name
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    # Explainer strip does not need boards.
    save(_plot_explainer_strip(), "004-scheduler-shape-guide.png")

    if not completed:
        return list(written)

    note = f"Source: {len(completed)}/{len(runs)} completed Exp 004 runs"

    for schedule_id in SCHEDULE_IDS:
        lr_figure = _plot_lr_components_for_schedule(completed, schedule_id)
        if lr_figure is not None:
            save(lr_figure, f"004-lr-{schedule_id}-seeds-100-101.png")

        train_figure = _plot_dual_seed_metric(
            completed,
            schedule_id,
            "train_acc",
            y_label="training accuracy (%)",
            title_prefix="Training accuracy",
            scale=100.0,
        )
        if train_figure is not None:
            save(train_figure, f"004-train-acc-{schedule_id}-seeds-100-101.png")

        val_figure = _plot_dual_seed_metric(
            completed,
            schedule_id,
            "val_acc",
            y_label="validation accuracy (%)",
            title_prefix="Validation accuracy",
            scale=100.0,
        )
        if val_figure is not None:
            save(val_figure, f"004-val-acc-{schedule_id}-seeds-100-101.png")

    train_changes = post_action_train_accuracy_changes(completed)
    post_action_figure = _plot_post_action_train_changes(train_changes)
    if post_action_figure is not None:
        save(
            post_action_figure,
            "004-post-action-train-acc-change-by-schedule.png",
        )

    figure, axis = plt.subplots(figsize=(10, 4.5))
    xs = list(range(len(SCHEDULE_IDS)))
    means_train = []
    means_val = []
    for schedule_id in SCHEDULE_IDS:
        subset = [run for run in completed if run["schedule_id"] == schedule_id]
        means_train.append(100.0 * _mean([float(run["final_train_acc"]) for run in subset]))
        means_val.append(100.0 * _mean([float(run["final_val_acc"]) for run in subset]))
        for run in subset:
            axis.scatter(
                SCHEDULE_IDS.index(schedule_id) - 0.12,
                100.0 * float(run["final_train_acc"]),
                color=SEED_COLORS.get(int(run["seed"]), "#888888"),
                s=28,
                zorder=3,
            )
            axis.scatter(
                SCHEDULE_IDS.index(schedule_id) + 0.12,
                100.0 * float(run["final_val_acc"]),
                color=SEED_COLORS.get(int(run["seed"]), "#888888"),
                s=28,
                marker="D",
                zorder=3,
            )
    axis.bar([x - 0.18 for x in xs], means_train, width=0.32, label="mean final train", color="#9bb7d4")
    axis.bar([x + 0.18 for x in xs], means_val, width=0.32, label="mean final val", color="#3568a8")
    axis.set_xticks(xs)
    axis.set_xticklabels([_short(schedule_id) for schedule_id in SCHEDULE_IDS], rotation=25, ha="right")
    axis.set_ylabel("accuracy (%)")
    axis.set_title(f"Final accuracy by LR schedule\n{note}")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    save(figure, "004-final-accuracy-by-schedule.png")

    return list(written)

if __name__ == "__main__":
    paths = generate_charts()
    print(f"Wrote {len(paths)} chart(s)")
    for path in paths:
        print(path)
