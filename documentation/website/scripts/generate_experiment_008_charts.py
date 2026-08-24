"""Generate charts for Experiment 008: CIFAR-10 adaptive meta-parameter search.

Measured figures (boards or snapshot):
- ranked trial validation accuracy, colored by starter
- peak vs final validation accuracy
- per-axis search grades
- start vs final parameter counts
- simulations vs executed actions by scheduler
- training and validation accuracy curves, colored by starter
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.train_cifar10_exp008_initial_package import combo_folder_name

SITE = Path(__file__).parents[1]
_RUNS_ROOT = SITE.parents[1] / "experiments" / "output" / "train_cifar10" / "runs"
DEFAULT_RUNS = _RUNS_ROOT / "exp008_cifar10_initial_package"
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-008-cifar10-initial-package.json"
_ALLOWED_SNAPSHOT_ROOT = (SITE / "data" / "experiments").resolve()
_ALLOWED_OUTPUT_ROOT = (SITE / "app" / "public" / "assets" / "experiments").resolve()
_ALLOWED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
SEARCH_JSON = "adaptive_search.json"

STARTER_ORDER = ("narrow", "base", "mid", "deep")
STARTER_COLORS = {
    "narrow": "#6b7280",
    "base": "#3568a8",
    "mid": "#0f766e",
    "deep": "#d18b2c",
}
SCHEDULER_MARKERS = {
    "always": "o",
    "slope_2deg": "^",
    "slope_3deg": "s",
}
AXIS_ORDER = (
    "starter",
    "epochs",
    "generations",
    "simulation_alg",
    "lr_schedule",
    "lr_alpha",
    "simulation_time",
    "simulation_epochs",
    "simulation_set_size",
    "simulation_scheduler",
)
ALG_LABELS = {
    "montecarlo": "MCTS",
    "greedy": "greedy",
    "sequential_halving_beam": "halving+beam",
    "ugape_deepen": "UGapE+deepen",
    "best_first": "best-first",
}
LR_LABELS = {
    "composed_exponential": "exponential",
    "composed_step": "step",
    "composed_cosine": "cosine",
}
SCHED_LABELS = {
    "always": "always",
    "slope_2deg": "2°",
    "slope_3deg": "3°",
}


def _resolve_under_allowed_root(path: Path, allowed_root: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("path must not contain '..'")
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(allowed_root) or resolved.is_relative_to(_ALLOWED_TEMP_ROOT):
        return resolved
    raise ValueError(
        f"path {resolved} is outside allowed roots {allowed_root} and {_ALLOWED_TEMP_ROOT}"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _axis_label(axis: str, value: object) -> str:
    text = str(value)
    if axis == "simulation_alg":
        return ALG_LABELS.get(text, text)
    if axis == "lr_schedule":
        return LR_LABELS.get(text, text)
    if axis == "simulation_scheduler":
        return SCHED_LABELS.get(text, text)
    if axis == "simulation_time":
        return str(int(float(text))) if float(text).is_integer() else text
    return text


def _load_board_run(run_dir: Path) -> dict[str, object] | None:
    main_path = run_dir / "board" / "main.json"
    metrics_path = run_dir / "board" / "metrics" / "training.json"
    if not main_path.is_file() or not metrics_path.is_file():
        return None
    main = json.loads(main_path.read_text(encoding="utf-8"))
    if main.get("status") != "completed":
        return None
    epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
    actions = [
        item["actionExecuted"]
        for item in main.get("generationTimeline", [])
        if item.get("actionExecuted")
    ]
    simulations_dir = run_dir / "board" / "simulations"
    simulations_ran = (
        len(list(simulations_dir.glob("simulation_gen_*.json")))
        if simulations_dir.is_dir()
        else 0
    )
    train_acc = [float(row["trainAcc"]) for row in epochs]
    val_acc = [float(row["valAcc"]) for row in epochs]
    param_count = [int(row.get("paramCount", 0)) for row in epochs]
    return {
        "status": main["status"],
        "elapsed_sec": float(main.get("trainingTimeElapsedSec") or 0),
        "started_on": main.get("experimentStartedOn"),
        "simulations_ran": simulations_ran,
        "actions": len(actions),
        "action_labels": [action["shortLabel"] for action in actions],
        "train_acc": train_acc,
        "val_acc": val_acc,
        "param_count": param_count,
        "final_train_acc": train_acc[-1] if train_acc else 0.0,
        "final_val_acc": val_acc[-1] if val_acc else 0.0,
        "peak_val_acc": max(val_acc) if val_acc else 0.0,
        "start_params": param_count[0] if param_count else 0,
        "final_params": param_count[-1] if param_count else 0,
    }


def load_search_and_runs(runs_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Join adaptive_search.json trials with completed board metrics."""
    search_path = runs_dir / SEARCH_JSON
    if not search_path.is_file():
        return {}, []
    search = json.loads(search_path.read_text(encoding="utf-8"))
    runs: list[dict[str, object]] = []
    for index, trial in enumerate(search.get("trials") or []):
        combo = dict(trial["combo"])
        seed = 100 + index
        folder = combo_folder_name(combo)
        board = _load_board_run(runs_dir / folder / f"seed_{seed}") or {}
        row = {
            "trial": index + 1,
            "seed": seed,
            "folder": folder,
            "combo": combo,
            "starter": combo["starter"],
            "simulation_alg": combo["simulation_alg"],
            "lr_schedule": combo["lr_schedule"],
            "simulation_scheduler": combo["simulation_scheduler"],
            "search_val_acc": float(trial["val_acc"]),
            "search_test_acc": float(trial["test_acc"]),
            **board,
        }
        runs.append(row)
    return search, runs


def write_snapshot(
    search: dict[str, object],
    runs: list[dict[str, object]],
    snapshot_path: Path,
    folder: str,
) -> None:
    resolved = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "008",
        "folder": folder,
        "iteration": search.get("iteration"),
        "max_iters": search.get("max_iters"),
        "unevaluated_count": search.get("unevaluated_count"),
        "pool_size": search.get("pool_size"),
        "best": search.get("best"),
        "grades": search.get("grades"),
        "probabilities": search.get("probabilities"),
        "runs": runs,
    }
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_runs_or_snapshot(
    runs_dir: Path, snapshot_path: Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    search, runs = load_search_and_runs(runs_dir)
    if runs:
        write_snapshot(search, runs, snapshot_path, runs_dir.name)
        return search, runs
    resolved_snapshot = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    if not resolved_snapshot.exists():
        return {}, []
    payload = json.loads(resolved_snapshot.read_text(encoding="utf-8"))
    search = {
        "iteration": payload.get("iteration"),
        "max_iters": payload.get("max_iters"),
        "unevaluated_count": payload.get("unevaluated_count"),
        "pool_size": payload.get("pool_size"),
        "best": payload.get("best"),
        "grades": payload.get("grades") or {},
        "probabilities": payload.get("probabilities") or {},
    }
    return search, list(payload.get("runs", []))


def plot_trial_accuracy(runs: list[dict[str, object]], output_dir: Path) -> None:
    ordered = sorted(runs, key=lambda row: float(row["search_val_acc"]))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(ordered))
    colors = [STARTER_COLORS[str(row["starter"])] for row in ordered]
    ax.barh(y, [float(row["search_val_acc"]) * 100 for row in ordered], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"t{row['trial']} {row['starter']} {SCHED_LABELS.get(str(row['simulation_scheduler']), str(row['simulation_scheduler']))}"
            for row in ordered
        ]
    )
    ax.set_xlabel("Peak validation accuracy (%)")
    ax.set_title("Peak validation accuracy by search trial")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STARTER_COLORS[name], label=name)
        for name in STARTER_ORDER
        if any(str(row["starter"]) == name for row in runs)
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "008-trial-val-acc.png", dpi=150)
    plt.close(fig)


def plot_peak_vs_final(runs: list[dict[str, object]], output_dir: Path) -> None:
    scored = [row for row in runs if "final_val_acc" in row]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for row in scored:
        starter = str(row["starter"])
        ax.scatter(
            float(row["search_val_acc"]) * 100,
            float(row["final_val_acc"]) * 100,
            color=STARTER_COLORS[starter],
            marker=SCHEDULER_MARKERS.get(str(row["simulation_scheduler"]), "o"),
            s=48,
            zorder=3,
        )
    lims = [35, 75]
    ax.plot(lims, lims, color="#888", linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Peak validation accuracy (%)")
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_title("Peak versus final validation accuracy")
    handles = [
        plt.Line2D([0], [0], color=STARTER_COLORS[name], marker="o", linestyle="", label=name)
        for name in STARTER_ORDER
        if any(str(row["starter"]) == name for row in scored)
    ]
    ax.legend(handles=handles)
    fig.tight_layout()
    fig.savefig(output_dir / "008-peak-vs-final-val.png", dpi=150)
    plt.close(fig)


def plot_axis_grades(search: dict[str, object], output_dir: Path) -> None:
    grades = search.get("grades") or {}
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.8))
    for ax, axis in zip(axes.ravel(), AXIS_ORDER):
        table = grades.get(axis) or {}
        labels = [_axis_label(axis, key) for key in table]
        values = [float(value) for value in table.values()]
        ax.barh(range(len(labels)), values, color="#3568a8")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.45, 0.75)
        ax.axvline(0.5, color="#888", linewidth=0.8)
        ax.set_title(axis.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Per-axis search grades after scored trials")
    fig.tight_layout()
    fig.savefig(output_dir / "008-axis-grades.png", dpi=150)
    plt.close(fig)


def plot_param_growth(runs: list[dict[str, object]], output_dir: Path) -> None:
    scored = [row for row in runs if "start_params" in row]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(scored))
    ax.bar(x - 0.2, [float(row["start_params"]) for row in scored], 0.4, label="start", color="#c5c9d0")
    ax.bar(x + 0.2, [float(row["final_params"]) for row in scored], 0.4, label="final", color="#3568a8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{row['trial']}" for row in scored])
    ax.set_ylabel("Parameter count")
    ax.set_xlabel("Search trial")
    ax.legend()
    ax.set_title("Start and final parameter counts by trial")
    fig.tight_layout()
    fig.savefig(output_dir / "008-param-growth.png", dpi=150)
    plt.close(fig)


def plot_search_activity(runs: list[dict[str, object]], output_dir: Path) -> None:
    scored = [row for row in runs if "simulations_ran" in row]
    by_sched: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scored:
        by_sched[str(row["simulation_scheduler"])].append(row)
    order = [key for key in ("always", "slope_2deg", "slope_3deg") if key in by_sched]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(order))
    sim_means = [_mean([float(row["simulations_ran"]) for row in by_sched[key]]) for key in order]
    act_means = [_mean([float(row["actions"]) for row in by_sched[key]]) for key in order]
    ax.bar(x - 0.2, sim_means, 0.4, label="simulations run", color="#d18b2c")
    ax.bar(x + 0.2, act_means, 0.4, label="actions executed", color="#3568a8")
    for i, key in enumerate(order):
        for row in by_sched[key]:
            ax.scatter(i - 0.2, float(row["simulations_ran"]), color="#444", s=18, zorder=3)
            ax.scatter(i + 0.2, float(row["actions"]), color="#444", s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SCHED_LABELS[key] for key in order])
    ax.set_ylabel("Count")
    ax.set_xlabel("Simulation scheduler")
    ax.legend()
    ax.set_title("Simulations run versus actions executed")
    fig.tight_layout()
    fig.savefig(output_dir / "008-search-activity-by-scheduler.png", dpi=150)
    plt.close(fig)


def _plot_accuracy_curves(
    runs: list[dict[str, object]],
    output_dir: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    scored = [row for row in runs if metric_key in row]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for row in scored:
        ys = [value * 100 for value in row[metric_key]]
        ax.plot(
            range(1, len(ys) + 1),
            ys,
            color=STARTER_COLORS[str(row["starter"])],
            alpha=0.8,
            linewidth=1.2,
            label=str(row["starter"]),
        )
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    ax.legend(seen.values(), seen.keys())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def plot_training_curves(runs: list[dict[str, object]], output_dir: Path) -> None:
    _plot_accuracy_curves(
        runs,
        output_dir,
        "train_acc",
        "Training accuracy (%)",
        "Training accuracy curves by starter",
        "008-training-curves.png",
    )


def plot_validation_curves(runs: list[dict[str, object]], output_dir: Path) -> None:
    _plot_accuracy_curves(
        runs,
        output_dir,
        "val_acc",
        "Validation accuracy (%)",
        "Validation accuracy curves by starter",
        "008-validation-curves.png",
    )


def plot_action_composition(runs: list[dict[str, object]], output_dir: Path) -> None:
    scored = [row for row in runs if "action_labels" in row]
    labels = sorted({label for row in scored for label in row["action_labels"]})
    if not labels:
        labels = ["(no actions)"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(scored))
    bottoms = np.zeros(len(scored))
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(labels):
        heights = [Counter(row.get("action_labels", []))[label] for row in scored]
        ax.bar(x, heights, bottom=bottoms, label=label, color=cmap(idx % 20))
        bottoms = bottoms + np.array(heights)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{row['trial']}" for row in scored])
    ax.set_ylabel("Executed actions")
    ax.set_xlabel("Search trial")
    ax.legend(fontsize=8, ncols=2)
    ax.set_title("Executed action mix by trial")
    fig.tight_layout()
    fig.savefig(output_dir / "008-action-composition.png", dpi=150)
    plt.close(fig)


def generate_charts(
    runs: list[dict[str, object]],
    output_dir: Path,
    search: dict[str, object] | None = None,
) -> None:
    plot_trial_accuracy(runs, output_dir)
    plot_peak_vs_final(runs, output_dir)
    if search:
        plot_axis_grades(search, output_dir)
    plot_param_growth(runs, output_dir)
    plot_search_activity(runs, output_dir)
    plot_action_composition(runs, output_dir)
    plot_training_curves(runs, output_dir)
    plot_validation_curves(runs, output_dir)


def main() -> None:
    runs_dir = DEFAULT_RUNS
    output_dir = _resolve_under_allowed_root(DEFAULT_OUTPUT, _ALLOWED_OUTPUT_ROOT)
    snapshot_path = DEFAULT_SNAPSHOT
    output_dir.mkdir(parents=True, exist_ok=True)
    search, runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    if not runs:
        print(
            f"No completed Exp 008 trials under {runs_dir}. "
            f"Snapshot empty or missing: {snapshot_path}"
        )
        return
    generate_charts(runs, output_dir, search=search)
    print(f"Wrote Exp 008 charts for {len(runs)} scored trials to {output_dir}")


if __name__ == "__main__":
    main()
