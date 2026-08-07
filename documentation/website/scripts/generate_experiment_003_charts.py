"""Generate charts for Experiment 003: simulation score accuracy metric."""

from __future__ import annotations

import json
from collections import defaultdict
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
    / "exp003_score_accuracy_metric"
)
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-003-score-accuracy-metric.json"

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
SEED_COLORS = {100: "#3568a8", 101: "#4f8a63", 102: "#d18b2c", 103: "#7a5a9a"}
SHORT_NAMES = {
    "medium_1conv_2linear": "med 1c+2l",
    "val_acc": "grade val",
    "train_acc": "grade train",
}


def _short(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
        )
    snapshot_path.write_text(json.dumps({"runs": compact}, indent=2), encoding="utf-8")


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
    completed = [run for run in runs if run["status"] == "completed"]
    note = f"Source: {len(completed)}/{len(runs)} completed Exp 003 runs"
    written: list[Path] = []

    def save(figure: plt.Figure, name: str) -> Path:
        path = output_dir / name
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
        return path

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
    save(figure, "003-final-accuracy-by-score-metric.png")

    # Dropout counts: the Exp 002 failure mode under test.
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
    figure.text(
        0.99,
        0.01,
        f"{note} · Exp 002 highlighted stacked dropout under val grading",
        ha="right",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save(figure, "003-dropout-actions-by-score-metric.png")

    # Action composition totals by grading mode.
    composition: dict[str, dict[str, int]] = {
        score_metric: defaultdict(int) for score_metric in SCORE_METRICS
    }
    for run in completed:
        for label in list(run["action_labels"]):
            composition[str(run["score_metric"])][str(label)] += 1
    type_list = sorted({name for score_metric in SCORE_METRICS for name in composition[score_metric]})
    if type_list:
        figure, axis = plt.subplots(figsize=(10.5, 4.8))
        width = 0.36
        short_types = [
            name.replace(" Action", "").replace("Add ", "+").replace("Delete ", "−")
            for name in type_list
        ]
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
        save(figure, "003-action-composition-by-score-metric.png")

    # Training curves: one panel per score × model.
    combos = [
        (score_metric, model)
        for score_metric in SCORE_METRICS
        for model in MODELS
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
    save(figure, "003-training-curves.png")

    return written


if __name__ == "__main__":
    paths = generate_charts()
    print(f"Wrote {len(paths)} chart(s)")
    for path in paths:
        print(path)
