"""Generate charts for Experiment 007: simulation-set generators.

Measured figures (boards or snapshot), one panel per simulation set size:
- final train/val accuracy by generator with seed markers
- seed scatter of final validation accuracy
- start vs final parameter counts by generator
- executed action mix by generator
- training-accuracy curves (faint seeds, bold mean) colored by generator

This grid uses sizes 100, 500, and 1000. Size 2000 boards are ignored.
"""

from __future__ import annotations

import json
import re
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
_RUNS_ROOT = SITE.parents[1] / "experiments" / "output" / "train_mnist" / "runs"
DEFAULT_RUNS = _RUNS_ROOT / "exp007_simulation_sets"
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-007-simulation-sets.json"
_ALLOWED_SNAPSHOT_ROOT = (SITE / "data" / "experiments").resolve()
_ALLOWED_OUTPUT_ROOT = (SITE / "app" / "public" / "assets" / "experiments").resolve()
_ALLOWED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()

SIMULATION_SET_SIZES = (100, 500, 1000)
GROUP_ORDER = (
    "protected",
    "moderate_difficulty",
    "kcenter",
    "el2n",
    "craig",
    "model_drift",
)
GROUP_LABELS = {key: key for key in GROUP_ORDER}
GROUP_COLORS = {
    "protected": "#6b7280",
    "moderate_difficulty": "#3568a8",
    "kcenter": "#4f8a63",
    "el2n": "#d18b2c",
    "craig": "#b45309",
    "model_drift": "#be185d",
}
_SIMSZ_RE = re.compile(r"_simsz(\d+)")


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


def _sim_size_from_board_or_path(main: dict, parts: tuple[str, ...]) -> int:
    params = main.get("trainingParameters") or {}
    raw = params.get("simulationSetSize")
    if raw is not None:
        return int(raw)
    if len(parts) >= 2:
        match = _SIMSZ_RE.search(parts[1])
        if match:
            return int(match.group(1))
    return 0


def _is_current_grid_run(run: dict[str, object]) -> bool:
    return (
        str(run.get("group_id")) in GROUP_ORDER
        and int(run.get("sim_size", 0)) in SIMULATION_SET_SIZES
    )


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    """Load board metrics for every group × seed run."""
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
        group_id = parts[0]
        if group_id not in GROUP_ORDER:
            continue
        main = json.loads(main_resolved.read_text(encoding="utf-8"))
        if main.get("status") != "completed":
            continue
        sim_size = _sim_size_from_board_or_path(main, parts)
        if sim_size not in SIMULATION_SET_SIZES:
            continue
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        actions = [
            item["actionExecuted"]["shortLabel"]
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        runs.append(
            {
                "group_id": group_id,
                "sim_size": sim_size,
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "action_labels": actions,
                "train_acc": [float(row["trainAcc"]) for row in epochs],
                "val_acc": [float(row["valAcc"]) for row in epochs],
                "param_count": [int(row.get("paramCount", 0)) for row in epochs],
                "final_train_acc": float(epochs[-1]["trainAcc"]) if epochs else 0.0,
                "final_val_acc": float(epochs[-1]["valAcc"]) if epochs else 0.0,
                "start_params": int(epochs[0].get("paramCount", 0)) if epochs else 0,
                "final_params": int(epochs[-1].get("paramCount", 0)) if epochs else 0,
            }
        )
    return runs


def write_snapshot(runs: list[dict[str, object]], snapshot_path: Path, folder: str) -> None:
    resolved = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": "007", "folder": folder, "runs": runs}
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
    return [run for run in payload.get("runs", []) if _is_current_grid_run(run)]


def _grouped(runs: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        by_group[str(run["group_id"])].append(run)
    return by_group


def _present_groups(runs: list[dict[str, object]]) -> tuple[str, ...]:
    present = {str(run["group_id"]) for run in runs}
    return tuple(group_id for group_id in GROUP_ORDER if group_id in present)


def _runs_for_size(runs: list[dict[str, object]], size: int) -> list[dict[str, object]]:
    return [run for run in runs if int(run["sim_size"]) == size]


def _size_panels(runs: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(size for size in SIMULATION_SET_SIZES if _runs_for_size(runs, size))


def _size_figure(runs: list[dict[str, object]], figsize_per_panel: float = 4.4):
    sizes = _size_panels(runs)
    if not sizes:
        return None, ()
    fig, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(figsize_per_panel * len(sizes), 4.6),
        sharey=True,
    )
    if len(sizes) == 1:
        axes = [axes]
    return fig, tuple(zip(sizes, axes))


def plot_final_accuracy(runs: list[dict[str, object]], output_dir: Path) -> None:
    fig, panels = _size_figure(runs)
    if fig is None:
        return
    width = 0.35
    groups = GROUP_ORDER
    x = np.arange(len(groups))
    for size, ax in panels:
        subset = _runs_for_size(runs, size)
        by_group = _grouped(subset)
        train_means = [
            _mean([float(r["final_train_acc"]) for r in by_group.get(g, [])]) * 100
            for g in groups
        ]
        val_means = [
            _mean([float(r["final_val_acc"]) for r in by_group.get(g, [])]) * 100
            for g in groups
        ]
        ax.bar(x - width / 2, train_means, width, label="train", color="#9db7d8")
        ax.bar(x + width / 2, val_means, width, label="val", color="#6f9e7d")
        for i, group_id in enumerate(groups):
            for run in by_group.get(group_id, []):
                ax.scatter(i - width / 2, float(run["final_train_acc"]) * 100, color="#444", s=18, zorder=3)
                ax.scatter(i + width / 2, float(run["final_val_acc"]) * 100, color="#444", s=18, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=25, ha="right")
        ax.set_title(f"size {size}")
        ax.legend()
    panels[0][1].set_ylabel("Final accuracy (%)")
    fig.suptitle("Final accuracy by simulation-set generator")
    fig.tight_layout()
    fig.savefig(output_dir / "007-final-accuracy-by-set.png", dpi=150)
    plt.close(fig)


def plot_param_growth(runs: list[dict[str, object]], output_dir: Path) -> None:
    fig, panels = _size_figure(runs)
    if fig is None:
        return
    width = 0.35
    groups = GROUP_ORDER
    x = np.arange(len(groups))
    for size, ax in panels:
        subset = _runs_for_size(runs, size)
        by_group = _grouped(subset)
        start_means = [_mean([float(r["start_params"]) for r in by_group.get(g, [])]) for g in groups]
        final_means = [_mean([float(r["final_params"]) for r in by_group.get(g, [])]) for g in groups]
        ax.bar(x - width / 2, start_means, width, label="start", color="#c5c9d0")
        ax.bar(x + width / 2, final_means, width, label="final", color="#3568a8")
        for i, group_id in enumerate(groups):
            for run in by_group.get(group_id, []):
                ax.scatter(i + width / 2, float(run["final_params"]), color="#444", s=18, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=25, ha="right")
        ax.set_title(f"size {size}")
        ax.legend()
    panels[0][1].set_ylabel("Parameter count")
    fig.suptitle("Parameter growth by simulation-set generator")
    fig.tight_layout()
    fig.savefig(output_dir / "007-param-growth-by-set.png", dpi=150)
    plt.close(fig)


def plot_action_composition(runs: list[dict[str, object]], output_dir: Path) -> None:
    fig, panels = _size_figure(runs)
    if fig is None:
        return
    labels = sorted({label for run in runs for label in run["action_labels"]})
    if not labels:
        labels = ["(no actions)"]
    groups = GROUP_ORDER
    x = np.arange(len(groups))
    cmap = plt.get_cmap("tab20")
    for size, ax in panels:
        subset = _runs_for_size(runs, size)
        by_group = _grouped(subset)
        bottoms = np.zeros(len(groups))
        for idx, label in enumerate(labels):
            heights = []
            for group_id in groups:
                group_runs = by_group.get(group_id, [])
                counts = [Counter(run["action_labels"])[label] for run in group_runs]
                heights.append(_mean(counts))
            ax.bar(x, heights, bottom=bottoms, label=label, color=cmap(idx % 20))
            bottoms = bottoms + np.array(heights)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=25, ha="right")
        ax.set_title(f"size {size}")
    panels[0][1].set_ylabel("Mean executed actions")
    panels[-1][1].legend(fontsize=8, ncols=2)
    fig.suptitle("Executed action mix by simulation-set generator")
    fig.tight_layout()
    fig.savefig(output_dir / "007-action-composition-by-set.png", dpi=150)
    plt.close(fig)


def plot_training_curves(runs: list[dict[str, object]], output_dir: Path) -> None:
    fig, panels = _size_figure(runs)
    if fig is None:
        return
    groups = GROUP_ORDER
    for size, ax in panels:
        subset = _runs_for_size(runs, size)
        by_group = _grouped(subset)
        for run in subset:
            group_id = str(run["group_id"])
            ys = [v * 100 for v in run["train_acc"]]
            ax.plot(
                range(1, len(ys) + 1),
                ys,
                color=GROUP_COLORS.get(group_id, "#333"),
                alpha=0.22,
                linewidth=1.0,
            )
        for group_id in groups:
            series = [run["train_acc"] for run in by_group.get(group_id, [])]
            if not series:
                continue
            length = min(len(row) for row in series)
            mean_ys = [
                _mean([float(row[index]) for row in series]) * 100 for index in range(length)
            ]
            ax.plot(
                range(1, length + 1),
                mean_ys,
                color=GROUP_COLORS.get(group_id, "#333"),
                linewidth=2.2,
                label=GROUP_LABELS.get(group_id, group_id),
            )
        ax.set_title(f"size {size}")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8, ncols=1)
    panels[0][1].set_ylabel("Training accuracy (%)")
    fig.suptitle("Training accuracy curves by simulation-set generator")
    fig.tight_layout()
    fig.savefig(output_dir / "007-training-curves.png", dpi=150)
    plt.close(fig)


def plot_seed_stability(runs: list[dict[str, object]], output_dir: Path) -> None:
    fig, panels = _size_figure(runs)
    if fig is None:
        return
    groups = GROUP_ORDER
    for size, ax in panels:
        subset = _runs_for_size(runs, size)
        by_group = _grouped(subset)
        for index, group_id in enumerate(groups):
            vals = [float(run["final_val_acc"]) * 100 for run in by_group.get(group_id, [])]
            if not vals:
                continue
            ax.scatter(
                [index] * len(vals),
                vals,
                color=GROUP_COLORS.get(group_id, "#333"),
                s=28,
                zorder=3,
            )
            ax.scatter([index], [_mean(vals)], color="#d97706", s=42, marker="D", zorder=4)
        ax.scatter([], [], color="#444", s=28, label="seed")
        ax.scatter([], [], color="#d97706", s=42, marker="D", label="mean")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=25, ha="right")
        ax.set_title(f"size {size}")
        ax.legend()
    panels[0][1].set_ylabel("Final validation accuracy (%)")
    fig.suptitle("Seed scatter of final validation accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "007-seed-stability-final-val.png", dpi=150)
    plt.close(fig)


def main() -> None:
    runs_dir = DEFAULT_RUNS
    output_dir = _resolve_under_allowed_root(DEFAULT_OUTPUT, _ALLOWED_OUTPUT_ROOT)
    snapshot_path = DEFAULT_SNAPSHOT
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    if not runs:
        print(
            f"No completed Exp 007 runs under {runs_dir}. "
            f"Snapshot empty or missing: {snapshot_path}"
        )
        return
    plot_final_accuracy(runs, output_dir)
    plot_seed_stability(runs, output_dir)
    plot_param_growth(runs, output_dir)
    plot_action_composition(runs, output_dir)
    plot_training_curves(runs, output_dir)
    print(f"Wrote Exp 007 charts for {len(runs)} completed runs to {output_dir}")


if __name__ == "__main__":
    main()
