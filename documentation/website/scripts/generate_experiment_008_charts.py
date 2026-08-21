"""Generate charts for Experiment 008: CIFAR-10 initial package.

Measured figures (boards or snapshot):
- final train/val accuracy by variant with seed markers
- start vs final parameter counts by variant
- executed action mix by variant
- simulations run vs actions executed (does 3° fire?)
- post-action train recovery after one generation
- training and validation accuracy curves colored by variant
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

SITE = Path(__file__).parents[1]
_RUNS_ROOT = SITE.parents[1] / "experiments" / "output" / "train_cifar10" / "runs"
DEFAULT_RUNS = _RUNS_ROOT / "exp008_cifar10_initial_package"
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-008-cifar10-initial-package.json"
_ALLOWED_SNAPSHOT_ROOT = (SITE / "data" / "experiments").resolve()
_ALLOWED_OUTPUT_ROOT = (SITE / "app" / "public" / "assets" / "experiments").resolve()
_ALLOWED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()

GROUP_ORDER = ("narrow", "base", "deep", "epochs20", "always", "fixed")
GROUP_LABELS = {
    "base": "base",
    "narrow": "narrow",
    "deep": "deep",
    "epochs20": "epochs20",
    "always": "always",
    "fixed": "fixed",
}
GROUP_COLORS = {
    "base": "#3568a8",
    "narrow": "#4f8a63",
    "deep": "#d18b2c",
    "epochs20": "#8b5cf6",
    "always": "#0f766e",
    "fixed": "#6b7280",
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


def _epochs_per_generation(epochs: list[dict], main: dict) -> int:
    params = main.get("trainingParameters") or {}
    configured = params.get("epochsPerGeneration")
    if configured:
        return int(configured)
    generations = [int(row.get("generation", 0)) for row in epochs]
    if not generations:
        return 10
    counts = Counter(generations)
    return max(counts.values())


def _post_action_changes(
    train_acc: list[float],
    actions: list[tuple[int, dict]],
    epochs_per_generation: int,
) -> tuple[list[float], list[float]]:
    immediate: list[float] = []
    recovered: list[float] = []
    for generation, _action in actions:
        end_g = (int(generation) + 1) * epochs_per_generation - 1
        start_next = end_g + 1
        end_next = (int(generation) + 2) * epochs_per_generation - 1
        if start_next < len(train_acc):
            immediate.append(100.0 * (train_acc[start_next] - train_acc[end_g]))
        if end_next < len(train_acc):
            recovered.append(100.0 * (train_acc[end_next] - train_acc[end_g]))
    return immediate, recovered


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics for every variant × seed run."""
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
        variant_id = parts[0]
        main = json.loads(main_resolved.read_text(encoding="utf-8"))
        if main.get("status") != "completed":
            continue
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        timeline_actions = [
            (item["generation"], item["actionExecuted"])
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        simulations_dir = main_resolved.parent / "simulations"
        simulations_ran = (
            len(list(simulations_dir.glob("simulation_gen_*.json")))
            if simulations_dir.is_dir()
            else 0
        )
        train_acc = [float(row["trainAcc"]) for row in epochs]
        val_acc = [float(row["valAcc"]) for row in epochs]
        epg = _epochs_per_generation(epochs, main)
        immediate, recovered = _post_action_changes(train_acc, timeline_actions, epg)
        runs.append(
            {
                "group_id": variant_id,
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "simulations_ran": simulations_ran,
                "actions": len(timeline_actions),
                "action_generations": [generation for generation, _ in timeline_actions],
                "action_labels": [action["shortLabel"] for _, action in timeline_actions],
                "epochs_per_generation": epg,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "param_count": [int(row.get("paramCount", 0)) for row in epochs],
                "final_train_acc": float(epochs[-1]["trainAcc"]) if epochs else 0.0,
                "final_val_acc": float(epochs[-1]["valAcc"]) if epochs else 0.0,
                "start_params": int(epochs[0].get("paramCount", 0)) if epochs else 0,
                "final_params": int(epochs[-1].get("paramCount", 0)) if epochs else 0,
                "immediate_post_action_train_changes": immediate,
                "post_action_train_changes": recovered,
            }
        )
    return runs


def write_snapshot(runs: list[dict[str, object]], snapshot_path: Path, folder: str) -> None:
    resolved = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "008", "folder": folder, "runs": runs}
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


def _grouped(runs: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        by_group[str(run["group_id"])].append(run)
    return by_group


def plot_final_accuracy(runs: list[dict[str, object]], output_dir: Path) -> None:
    by_group = _grouped(runs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.35
    train_means = [_mean([float(r["final_train_acc"]) for r in by_group.get(g, [])]) * 100 for g in GROUP_ORDER]
    val_means = [_mean([float(r["final_val_acc"]) for r in by_group.get(g, [])]) * 100 for g in GROUP_ORDER]
    ax.bar(x - width / 2, train_means, width, label="train", color="#9db7d8")
    ax.bar(x + width / 2, val_means, width, label="val", color="#6f9e7d")
    for i, group_id in enumerate(GROUP_ORDER):
        for run in by_group.get(group_id, []):
            ax.scatter(i - width / 2, float(run["final_train_acc"]) * 100, color="#444", s=18, zorder=3)
            ax.scatter(i + width / 2, float(run["final_val_acc"]) * 100, color="#444", s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("Final accuracy (%)")
    ax.set_xlabel("CIFAR-10 package variant")
    ax.legend()
    ax.set_title("Final accuracy by CIFAR-10 package variant")
    fig.tight_layout()
    fig.savefig(output_dir / "008-final-accuracy-by-variant.png", dpi=150)
    plt.close(fig)


def plot_param_growth(runs: list[dict[str, object]], output_dir: Path) -> None:
    by_group = _grouped(runs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.35
    start_means = [_mean([float(r["start_params"]) for r in by_group.get(g, [])]) for g in GROUP_ORDER]
    final_means = [_mean([float(r["final_params"]) for r in by_group.get(g, [])]) for g in GROUP_ORDER]
    ax.bar(x - width / 2, start_means, width, label="start", color="#c5c9d0")
    ax.bar(x + width / 2, final_means, width, label="final", color="#3568a8")
    for i, group_id in enumerate(GROUP_ORDER):
        for run in by_group.get(group_id, []):
            ax.scatter(i + width / 2, float(run["final_params"]), color="#444", s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("Parameter count")
    ax.set_xlabel("CIFAR-10 package variant")
    ax.legend()
    ax.set_title("Parameter growth by CIFAR-10 package variant")
    fig.tight_layout()
    fig.savefig(output_dir / "008-param-growth-by-variant.png", dpi=150)
    plt.close(fig)


def plot_action_composition(runs: list[dict[str, object]], output_dir: Path) -> None:
    by_group = _grouped(runs)
    labels = sorted({label for run in runs for label in run["action_labels"]})
    if not labels:
        labels = ["(no actions)"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(GROUP_ORDER))
    bottoms = np.zeros(len(GROUP_ORDER))
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(labels):
        heights = []
        for group_id in GROUP_ORDER:
            group_runs = by_group.get(group_id, [])
            if not group_runs:
                heights.append(0.0)
                continue
            counts = [Counter(run["action_labels"])[label] for run in group_runs]
            heights.append(_mean(counts))
        ax.bar(x, heights, bottom=bottoms, label=label, color=cmap(idx % 20))
        bottoms = bottoms + np.array(heights)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("Mean executed actions")
    ax.set_xlabel("CIFAR-10 package variant")
    ax.legend(fontsize=8, ncols=2)
    ax.set_title("Executed action mix by CIFAR-10 package variant")
    fig.tight_layout()
    fig.savefig(output_dir / "008-action-composition-by-variant.png", dpi=150)
    plt.close(fig)


def plot_search_activity(runs: list[dict[str, object]], output_dir: Path) -> None:
    by_group = _grouped(runs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.35
    sim_means = [_mean([float(r.get("simulations_ran", 0)) for r in by_group.get(g, [])]) for g in GROUP_ORDER]
    action_means = [_mean([float(r.get("actions", 0)) for r in by_group.get(g, [])]) for g in GROUP_ORDER]
    ax.bar(x - width / 2, sim_means, width, label="simulations run", color="#d18b2c")
    ax.bar(x + width / 2, action_means, width, label="actions executed", color="#3568a8")
    for i, group_id in enumerate(GROUP_ORDER):
        for run in by_group.get(group_id, []):
            ax.scatter(i - width / 2, float(run.get("simulations_ran", 0)), color="#444", s=18, zorder=3)
            ax.scatter(i + width / 2, float(run.get("actions", 0)), color="#444", s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("Mean count")
    ax.set_xlabel("CIFAR-10 package variant")
    ax.legend()
    ax.set_title("Simulations run versus actions executed")
    fig.tight_layout()
    fig.savefig(output_dir / "008-search-activity-by-variant.png", dpi=150)
    plt.close(fig)


def plot_post_action_recovery(runs: list[dict[str, object]], output_dir: Path) -> None:
    by_group = _grouped(runs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.35
    immediate_means = []
    recovered_means = []
    for group_id in GROUP_ORDER:
        immediate = [
            value
            for run in by_group.get(group_id, [])
            for value in run.get("immediate_post_action_train_changes", [])
        ]
        recovered = [
            value
            for run in by_group.get(group_id, [])
            for value in run.get("post_action_train_changes", [])
        ]
        immediate_means.append(_mean(immediate))
        recovered_means.append(_mean(recovered))
    ax.bar(x - width / 2, immediate_means, width, label="next epoch", color="#d18b2c")
    ax.bar(x + width / 2, recovered_means, width, label="after one generation", color="#3568a8")
    ax.axhline(0.0, color="#888", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_ylabel("Train-accuracy change (percentage points)")
    ax.set_xlabel("CIFAR-10 package variant")
    ax.legend()
    ax.set_title("Train accuracy change after architecture actions")
    fig.tight_layout()
    fig.savefig(output_dir / "008-post-action-recovery-by-variant.png", dpi=150)
    plt.close(fig)


def _plot_accuracy_curves(
    runs: list[dict[str, object]],
    output_dir: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in runs:
        group_id = str(run["group_id"])
        ys = [v * 100 for v in run[metric_key]]
        ax.plot(
            range(1, len(ys) + 1),
            ys,
            color=GROUP_COLORS.get(group_id, "#333"),
            alpha=0.75,
            linewidth=1.2,
            label=GROUP_LABELS.get(group_id, group_id),
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
        "Training accuracy curves by CIFAR-10 package variant",
        "008-training-curves.png",
    )


def plot_validation_curves(runs: list[dict[str, object]], output_dir: Path) -> None:
    _plot_accuracy_curves(
        runs,
        output_dir,
        "val_acc",
        "Validation accuracy (%)",
        "Validation accuracy curves by CIFAR-10 package variant",
        "008-validation-curves.png",
    )


def generate_charts(
    runs: list[dict[str, object]],
    output_dir: Path,
) -> None:
    plot_final_accuracy(runs, output_dir)
    plot_param_growth(runs, output_dir)
    plot_action_composition(runs, output_dir)
    plot_search_activity(runs, output_dir)
    plot_post_action_recovery(runs, output_dir)
    plot_training_curves(runs, output_dir)
    plot_validation_curves(runs, output_dir)


def main() -> None:
    runs_dir = DEFAULT_RUNS
    output_dir = _resolve_under_allowed_root(DEFAULT_OUTPUT, _ALLOWED_OUTPUT_ROOT)
    snapshot_path = DEFAULT_SNAPSHOT
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    if not runs:
        print(
            f"No completed Exp 008 runs under {runs_dir}. "
            f"Snapshot empty or missing: {snapshot_path}"
        )
        return
    generate_charts(runs, output_dir)
    print(f"Wrote Exp 008 charts for {len(runs)} completed runs to {output_dir}")


if __name__ == "__main__":
    main()
