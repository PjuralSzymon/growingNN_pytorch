"""Generate charts for Experiment 005: simulation algorithm comparison.

Measured figures (boards or snapshot):
- combined and per-starter final train/val accuracy (gray seed markers)
- seed scatter + purple composite score (stable / better on the right)
- first-action gain share with accuracy-leader highlights
- action-type mix (more diversity on the right)
- immediate vs recovered train change after actions
- training and validation history grids plus mean±std best-fit grids
- look-ahead vs depth-1 group mean accuracy
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
_RUNS_ROOT = SITE.parents[1] / "experiments" / "output" / "train_mnist" / "runs"
DEFAULT_RUNS = _RUNS_ROOT / "exp005_simulation_algorithms"
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-005-simulation-algorithms.json"
_ALLOWED_SNAPSHOT_ROOT = (SITE / "data" / "experiments").resolve()
_ALLOWED_OUTPUT_ROOT = (SITE / "app" / "public" / "assets" / "experiments").resolve()
_ALLOWED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()

ALG_ORDER = (
    "montecarlo",
    "greedy",
    "random",
    "sequential_halving",
    "ugape",
    "successive_rejects",
    "beam_search",
    "best_first",
    "shot",
    "sequential_halving_beam",
    "ugape_deepen",
    "progressive_widening",
    "hierarchical_search",
)
ALG_LABELS = {
    "montecarlo": "MCTS",
    "greedy": "greedy",
    "random": "random",
    "sequential_halving": "seq. halving",
    "ugape": "UGapE",
    "successive_rejects": "succ. rejects",
    "beam_search": "beam",
    "best_first": "best-first",
    "shot": "SHOT",
    "sequential_halving_beam": "halving+beam",
    "ugape_deepen": "UGapE+deepen",
    "progressive_widening": "prog. widen",
    "hierarchical_search": "hierarchical",
}
# Kept only for optional filtered panels; MCTS is NOT depth-1-only.
DEPTH1_ALGS = (
    "greedy",
    "random",
    "sequential_halving",
    "ugape",
    "successive_rejects",
)
LOOKAHEAD_ALGS = (
    "montecarlo",
    "beam_search",
    "best_first",
    "shot",
    "sequential_halving_beam",
    "ugape_deepen",
    "progressive_widening",
    "hierarchical_search",
)
MODEL_ORDER = (
    "big",
    "medium_1conv_2linear",
)
MODEL_LABELS = {
    "big": "big",
    "medium_1conv_2linear": "medium",
}
EPOCHS_PER_GENERATION = 10


def _parse_alg_and_model(parts: tuple[str, ...]) -> tuple[str, str]:
    """Return (alg_id, model_name) from run-relative path parts."""
    if len(parts) >= 4 and parts[1] in MODEL_ORDER:
        return parts[0], parts[1]
    return parts[0], "big"


def _short_action_name(name: str) -> str:
    return (
        str(name)
        .removeprefix("Add ")
        .removesuffix(" Action")
        .removesuffix(" Layer")
        .strip()
    )


def _model_short(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


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


def _sample_variance(values: list[float]) -> float:
    """Population variance over the listed seed values (0 if fewer than 2)."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _short(alg_id: str) -> str:
    return ALG_LABELS.get(alg_id, alg_id)


def _jaccard_counters(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    intersection = sum(min(left[key], right[key]) for key in keys)
    union = sum(max(left[key], right[key]) for key in keys)
    return intersection / union if union else 1.0


def _mean_pairwise_jaccard(bags: list[Counter[str]]) -> float | None:
    if len(bags) < 2:
        return None
    scores: list[float] = []
    for index, left in enumerate(bags):
        for right in bags[index + 1 :]:
            scores.append(_jaccard_counters(left, right))
    return _mean(scores)


def _linear_best_fit(ys: list[float]) -> tuple[list[float], float, float]:
    """Return fitted y values, slope, and intercept for y ~ a*x + b."""
    n = len(ys)
    if n == 0:
        return [], 0.0, 0.0
    xs = list(range(n))
    x_mean = _mean([float(x) for x in xs])
    y_mean = _mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return list(ys), 0.0, y_mean
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    fitted = [slope * x + intercept for x in xs]
    return fitted, slope, intercept


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
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
        alg_id, model_name = _parse_alg_and_model(parts)
        main = json.loads(main_resolved.read_text(encoding="utf-8"))
        epochs = json.loads(metrics_path.read_text(encoding="utf-8"))["epochs"]
        actions = [
            (item["generation"], item["actionExecuted"])
            for item in main.get("generationTimeline", [])
            if item.get("actionExecuted")
        ]
        train_acc = [float(row["trainAcc"]) for row in epochs]
        val_acc = [float(row["valAcc"]) for row in epochs]
        param_counts = [int(row["paramCount"]) for row in epochs]
        recovered_changes: list[float] = []
        immediate_changes: list[float] = []
        order_train_gains: list[float] = []
        order_val_gains: list[float] = []
        typed_train_gains: list[tuple[str, float]] = []
        typed_val_gains: list[tuple[str, float]] = []
        for generation, action in actions:
            label = str(action["shortLabel"])
            end_g = (int(generation) + 1) * EPOCHS_PER_GENERATION - 1
            start_next = end_g + 1
            end_next = (int(generation) + 2) * EPOCHS_PER_GENERATION - 1
            if start_next < len(train_acc):
                immediate_changes.append(100.0 * (train_acc[start_next] - train_acc[end_g]))
            if end_next < len(train_acc):
                train_gain = 100.0 * (train_acc[end_next] - train_acc[end_g])
                val_gain = 100.0 * (val_acc[end_next] - val_acc[end_g])
                recovered_changes.append(train_gain)
                order_train_gains.append(train_gain)
                order_val_gains.append(val_gain)
                typed_train_gains.append((label, train_gain))
                typed_val_gains.append((label, val_gain))
        runs.append(
            {
                "alg_id": alg_id,
                "model_name": model_name,
                "seed": int(parts[-1].removeprefix("seed_")),
                "status": main["status"],
                "elapsed_sec": main.get("trainingTimeElapsedSec"),
                "actions": len(actions),
                "action_generations": [generation for generation, _ in actions],
                "action_labels": [action["shortLabel"] for _, action in actions],
                "train_acc": train_acc,
                "val_acc": val_acc,
                "param_counts": param_counts,
                "final_param_count": param_counts[-1] if param_counts else 0,
                "final_train_acc": float(epochs[-1]["trainAcc"]) if epochs else 0.0,
                "final_val_acc": float(epochs[-1]["valAcc"]) if epochs else 0.0,
                "post_action_train_changes": recovered_changes,
                "immediate_post_action_train_changes": immediate_changes,
                "action_order_train_gains": order_train_gains,
                "action_order_val_gains": order_val_gains,
                "typed_train_gains": typed_train_gains,
                "typed_val_gains": typed_val_gains,
            }
        )
    return runs


def completed_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [run for run in runs if run.get("status") == "completed"]


def _grid_is_partial(runs: list[dict[str, object]]) -> bool:
    completed = completed_runs(runs)
    by_cell: dict[tuple[str, str], set[int]] = defaultdict(set)
    for run in completed:
        by_cell[(str(run["alg_id"]), str(run.get("model_name", "big")))].add(int(run["seed"]))
    return any(
        len(by_cell.get((alg_id, model_name), set())) < 5
        for alg_id in ALG_ORDER
        for model_name in MODEL_ORDER
    )


def write_snapshot(runs: list[dict[str, object]], snapshot_path: Path, folder: str) -> None:
    resolved = _resolve_under_allowed_root(snapshot_path, _ALLOWED_SNAPSHOT_ROOT)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "005",
        "folder": folder,
        "partial": _grid_is_partial(runs),
        "runs": runs,
    }
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


def present_alg_ids(runs: list[dict[str, object]]) -> list[str]:
    present = {str(run["alg_id"]) for run in runs}
    return [alg_id for alg_id in ALG_ORDER if alg_id in present]


def sort_algs_by(
    algs: list[str],
    completed: list[dict[str, object]],
    key_fn,
    *,
    reverse: bool = False,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for alg_id in algs:
        subset = [run for run in completed if run["alg_id"] == alg_id]
        scored.append((float(key_fn(subset)), alg_id))
    scored.sort(key=lambda item: item[0], reverse=reverse)
    return [alg_id for _, alg_id in scored]


def generate_charts(
    runs_dir: Path = DEFAULT_RUNS,
    output_dir: Path = DEFAULT_OUTPUT,
    snapshot_path: Path = DEFAULT_SNAPSHOT,
) -> list[Path]:
    from matplotlib.lines import Line2D

    runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    all_completed = completed_runs(runs)
    resolved_output = _resolve_under_allowed_root(output_dir, _ALLOWED_OUTPUT_ROOT)
    resolved_output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
    written: list[Path] = []

    def save(figure: plt.Figure, name: str) -> None:
        path = resolved_output / name
        figure.savefig(path)
        plt.close(figure)
        written.append(path)

    if not all_completed:
        return list(written)

    def model_runs(model_name: str) -> list[dict[str, object]]:
        return [run for run in all_completed if str(run.get("model_name", "big")) == model_name]

    def alg_subset(pool: list[dict[str, object]], alg_id: str) -> list[dict[str, object]]:
        return [run for run in pool if run["alg_id"] == alg_id]

    def first_action_share(subset: list[dict[str, object]]) -> float | None:
        positive_total = 0.0
        positive_first = 0.0
        for run in subset:
            for order, gain in enumerate(list(run.get("action_order_val_gains", [])), start=1):
                value = float(gain)
                if value <= 0:
                    continue
                positive_total += value
                if order == 1:
                    positive_first += value
        if positive_total <= 0:
            return None
        return 100.0 * positive_first / positive_total

    def unique_action_types(subset: list[dict[str, object]]) -> float:
        scores = [
            float(len({_short_action_name(label) for label in list(run.get("action_labels", []))}))
            for run in subset
        ]
        return _mean(scores)

    algs_present = present_alg_ids(all_completed)
    algs_by_train = sort_algs_by(
        algs_present,
        all_completed,
        lambda subset: _mean([float(run["final_train_acc"]) for run in subset]),
        reverse=False,
    )
    figure, axis = plt.subplots(figsize=(max(10, 1.35 * len(algs_by_train) + 2), 5.0))
    xs = list(range(len(algs_by_train)))
    width = 0.2
    big_train, big_val, med_train, med_val = [], [], [], []
    for alg_id in algs_by_train:
        big = alg_subset(model_runs("big"), alg_id)
        med = alg_subset(model_runs("medium_1conv_2linear"), alg_id)
        big_train.append(100.0 * _mean([float(r["final_train_acc"]) for r in big]) if big else float("nan"))
        big_val.append(100.0 * _mean([float(r["final_val_acc"]) for r in big]) if big else float("nan"))
        med_train.append(100.0 * _mean([float(r["final_train_acc"]) for r in med]) if med else float("nan"))
        med_val.append(100.0 * _mean([float(r["final_val_acc"]) for r in med]) if med else float("nan"))
    axis.bar([x - 1.5 * width for x in xs], big_train, width=width, label="big train", color="#b7c9dc")
    axis.bar([x - 0.5 * width for x in xs], big_val, width=width, label="big val", color="#3568a8")
    axis.bar([x + 0.5 * width for x in xs], med_train, width=width, label="medium train", color="#c9d9b7")
    axis.bar([x + 1.5 * width for x in xs], med_val, width=width, label="medium val", color="#4f8a63")
    axis.set_xticks(xs)
    axis.set_xticklabels([_short(alg_id) for alg_id in algs_by_train], rotation=20, ha="right")
    axis.set_ylabel("accuracy (%)")
    axis.set_title(
        f"Final accuracy by algorithm (big + medium)\n"
        f"{len(all_completed)} completed runs · left = lower mean train · right = higher mean train"
    )
    axis.legend(fontsize=8, ncol=2)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    save(figure, "005-final-accuracy-combined.png")

    for model_name in MODEL_ORDER:
        pool = model_runs(model_name)
        model_algs = present_alg_ids(pool)
        if not model_algs:
            continue
        model_tag = _model_short(model_name)
        suffix = "" if model_name == "big" else f"-{model_tag}"
        ordered = sort_algs_by(
            model_algs,
            pool,
            lambda subset: _mean([float(run["final_train_acc"]) for run in subset]),
            reverse=False,
        )
        figure, axis = plt.subplots(figsize=(max(8, 1.1 * len(ordered) + 2), 4.8))
        xs = list(range(len(ordered)))
        means_train, means_val = [], []
        for index, alg_id in enumerate(ordered):
            subset = alg_subset(pool, alg_id)
            means_train.append(100.0 * _mean([float(run["final_train_acc"]) for run in subset]))
            means_val.append(100.0 * _mean([float(run["final_val_acc"]) for run in subset]))
            for run in subset:
                axis.scatter(index - 0.12, 100.0 * float(run["final_train_acc"]), color="#777777", s=28, zorder=3)
                axis.scatter(index + 0.12, 100.0 * float(run["final_val_acc"]), color="#555555", s=28, marker="D", zorder=3)
        # Color scheme: blue = big, green = medium (same as combined chart).
        train_color = "#9bb7d4" if model_name == "big" else "#c9d9b7"
        val_color = "#3568a8" if model_name == "big" else "#4f8a63"
        axis.bar([x - 0.18 for x in xs], means_train, width=0.32, label="mean final train", color=train_color)
        axis.bar([x + 0.18 for x in xs], means_val, width=0.32, label="mean final val", color=val_color)
        axis.set_xticks(xs)
        axis.set_xticklabels([_short(alg_id) for alg_id in ordered], rotation=20, ha="right")
        axis.set_ylabel("accuracy (%)")
        axis.set_title(
            f"Final accuracy ({model_tag} starter, n={len(pool)})\n"
            "left = lower mean train · right = higher mean train"
        )
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, f"005-final-accuracy-by-algorithm{suffix}.png")

    def write_stability_panel(pool: list[dict[str, object]], tag: str, filename_suffix: str) -> None:
        model_algs = present_alg_ids(pool)
        if not model_algs:
            return
        algs_by_var = sort_algs_by(
            model_algs,
            pool,
            lambda subset: _sample_variance([100.0 * float(run["final_val_acc"]) for run in subset]),
            reverse=True,
        )
        figure, axis = plt.subplots(figsize=(max(8, 1.1 * len(algs_by_var) + 2), 4.2))
        for index, alg_id in enumerate(algs_by_var):
            subset = alg_subset(pool, alg_id)
            for run in subset:
                color = (
                    "#3568a8"
                    if str(run.get("model_name", "big")) == "big"
                    else "#4f8a63"
                )
                axis.scatter(
                    [index],
                    [100.0 * float(run["final_val_acc"])],
                    s=36,
                    color=color,
                    zorder=3,
                )
            vals = [100.0 * float(run["final_val_acc"]) for run in subset]
            if vals:
                axis.hlines(_mean(vals), index - 0.25, index + 0.25, colors="#d18b2c", linewidths=2)
        axis.set_xticks(list(range(len(algs_by_var))))
        axis.set_xticklabels([_short(alg_id) for alg_id in algs_by_var], rotation=20, ha="right")
        axis.set_ylabel("final validation accuracy (%)")
        axis.set_title(
            f"Final validation seeds ({tag})\n"
            "blue = big · green = medium · left = higher variance · right = tighter · orange = mean"
        )
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, f"005-seed-stability-final-val{filename_suffix}.png")

        composite_rows: list[tuple[float, str, int]] = []
        for alg_id in model_algs:
            subset = alg_subset(pool, alg_id)
            if len(subset) < 2:
                continue
            vals = [100.0 * float(run["final_val_acc"]) for run in subset]
            mean_val = _mean(vals)
            variance = _sample_variance(vals)
            composite_rows.append((mean_val - 0.15 * (variance ** 0.5), alg_id, len(subset)))
        composite_rows.sort(key=lambda item: item[0])
        if not composite_rows:
            return
        figure, axis = plt.subplots(figsize=(max(8, 1.1 * len(composite_rows) + 2), 4.4))
        scores = [row[0] for row in composite_rows]
        axis.bar(range(len(composite_rows)), scores, color="#7a6bb5")
        for index, (score, _alg_id, n_seeds) in enumerate(composite_rows):
            axis.text(index, score, f"n={n_seeds}", ha="center", va="bottom", fontsize=7)
        axis.set_xticks(list(range(len(composite_rows))))
        axis.set_xticklabels([_short(row[1]) for row in composite_rows], rotation=20, ha="right")
        axis.set_ylabel("composite (mean val - 0.15 * sqrt(var))")
        axis.set_title(
            f"Composite score ({tag})\nleft = worse · right = better · purple = pooled score (not starter-colored)"
        )
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, f"005-composite-score{filename_suffix}.png")

    write_stability_panel(all_completed, "all starters pooled", "")
    write_stability_panel(model_runs("medium_1conv_2linear"), "medium starter only", "-medium")
    write_stability_panel(model_runs("big"), "big starter only", "-big")

    shared_rows: list[tuple[str, float, float]] = []
    for alg_id in ALG_ORDER:
        big = [100.0 * float(r["final_train_acc"]) for r in model_runs("big") if r["alg_id"] == alg_id]
        med = [
            100.0 * float(r["final_train_acc"])
            for r in model_runs("medium_1conv_2linear")
            if r["alg_id"] == alg_id
        ]
        if len(big) >= 3 and len(med) >= 3:
            shared_rows.append((alg_id, _mean(big), _mean(med)))
    shared_rows.sort(key=lambda item: abs(item[2] - item[1]), reverse=True)
    if shared_rows:
        figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(shared_rows) + 2), 4.6))
        xs = list(range(len(shared_rows)))
        axis.bar([x - 0.18 for x in xs], [row[1] for row in shared_rows], width=0.32, label="big", color="#3568a8")
        axis.bar([x + 0.18 for x in xs], [row[2] for row in shared_rows], width=0.32, label="medium", color="#4f8a63")
        axis.set_xticks(xs)
        axis.set_xticklabels([_short(row[0]) for row in shared_rows], rotation=20, ha="right")
        axis.set_ylabel("mean final training accuracy (%)")
        axis.set_title(
            "Mean final training accuracy: big vs medium\n"
            "sorted by absolute starter gap · left = larger gap · right = more consistent across model sizes"
        )
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, "005-final-train-big-vs-medium-by-algorithm.png")

    share_rows: list[tuple[str, float]] = []
    for alg_id in present_alg_ids(all_completed):
        pooled = first_action_share(alg_subset(all_completed, alg_id))
        if pooled is None:
            continue
        share_rows.append((alg_id, pooled))
    share_rows.sort(key=lambda item: item[1], reverse=True)
    # Highlight the current top-3 mean-train leaders from final accuracy.
    accuracy_leaders = sort_algs_by(
        present_alg_ids(all_completed),
        all_completed,
        lambda subset: _mean([float(run["final_train_acc"]) for run in subset]),
        reverse=True,
    )[:3]
    if share_rows:
        figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(share_rows) + 2), 4.4))
        xs = list(range(len(share_rows)))
        colors = [
            "#d18b2c" if row[0] in accuracy_leaders else "#6b7280"
            for row in share_rows
        ]
        axis.bar(xs, [row[1] for row in share_rows], color=colors)
        for index, (alg_id, _share) in enumerate(share_rows):
            if alg_id in accuracy_leaders:
                axis.text(
                    index,
                    share_rows[index][1],
                    "top train",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#8a5a14",
                )
        axis.set_xticks(xs)
        axis.set_xticklabels([_short(row[0]) for row in share_rows], rotation=20, ha="right")
        axis.set_ylabel("share of positive recovered val gain from action 1 (%)")
        axis.set_title(
            "How much of the useful gain comes from the first live action\n"
            "left = first action does almost everything · right = later actions also matter\n"
            "orange bars = current top-3 by mean final train"
        )
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, "005-first-action-gain-share-by-algorithm.png")

    for model_name in ("__all__",) + MODEL_ORDER:
        pool = all_completed if model_name == "__all__" else model_runs(model_name)
        model_algs = present_alg_ids(pool)
        if not model_algs:
            continue
        ordered = sort_algs_by(model_algs, pool, unique_action_types, reverse=False)
        composition: dict[str, Counter[str]] = {alg_id: Counter() for alg_id in ordered}
        for run in pool:
            composition[str(run["alg_id"])].update(
                [_short_action_name(label) for label in list(run.get("action_labels", []))]
            )
        type_names = sorted(
            {name for counter in composition.values() for name in counter},
            key=lambda name: -sum(composition[alg_id][name] for alg_id in ordered),
        )
        if not type_names:
            continue
        if model_name == "__all__":
            tag = "all starters"
            suffix = ""
        else:
            tag = _model_short(model_name)
            suffix = f"-{tag}"
        figure, axis = plt.subplots(figsize=(max(9, 1.2 * len(ordered) + 2), 4.8))
        bottoms = [0.0] * len(ordered)
        colors = plt.cm.tab20.colors
        for type_index, type_name in enumerate(type_names):
            heights = [float(composition[alg_id][type_name]) for alg_id in ordered]
            axis.bar(
                range(len(ordered)),
                heights,
                bottom=bottoms,
                label=type_name,
                color=colors[type_index % len(colors)],
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        axis.set_xticks(list(range(len(ordered))))
        axis.set_xticklabels([_short(alg_id) for alg_id in ordered], rotation=20, ha="right")
        axis.set_ylabel("executed action count across completed seeds")
        axis.set_title(
            f"Action-type mix ({tag})\n"
            "left = mostly one action type · right = uses more action types"
        )
        axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, f"005-action-composition-by-algorithm{suffix}.png")

    ordered = sort_algs_by(
        present_alg_ids(all_completed),
        all_completed,
        lambda subset: _mean(
            [float(v) for run in subset for v in list(run.get("post_action_train_changes", []))]
        ),
        reverse=False,
    )
    figure, axis = plt.subplots(figsize=(max(8, 1.1 * len(ordered) + 2), 4.4))
    xs = list(range(len(ordered)))
    immediate_means = []
    recovered_means = []
    for alg_id in ordered:
        subset = alg_subset(all_completed, alg_id)
        immediate_means.append(
            _mean([float(v) for run in subset for v in list(run.get("immediate_post_action_train_changes", []))])
        )
        recovered_means.append(
            _mean([float(v) for run in subset for v in list(run.get("post_action_train_changes", []))])
        )
    axis.bar([x - 0.18 for x in xs], immediate_means, width=0.32, label="mean immediate (next epoch)", color="#c47b5a")
    axis.bar([x + 0.18 for x in xs], recovered_means, width=0.32, label="mean after 1 generation", color="#3568a8")
    axis.axhline(0.0, color="#666666", linewidth=1.0)
    axis.set_xticks(xs)
    axis.set_xticklabels([_short(alg_id) for alg_id in ordered], rotation=20, ha="right")
    axis.set_ylabel("train accuracy change (percentage points)")
    axis.set_title(
        "Training-accuracy change after an architecture action\n"
        "immediate next epoch vs after one recovery generation · left = weaker recovery"
    )
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    save(figure, "005-action-impact-immediate-vs-recovered.png")

    typed_vals: dict[str, list[float]] = defaultdict(list)
    for run in all_completed:
        for label, gain in list(run.get("typed_val_gains", [])):
            typed_vals[_short_action_name(str(label))].append(float(gain))
    type_names = sorted(typed_vals, key=lambda name: -_mean(typed_vals[name]))
    if type_names:
        figure, axes = plt.subplots(1, 2, figsize=(11.5, max(3.8, 0.35 * len(type_names) + 1.5)))
        y = list(range(len(type_names)))
        means = [_mean(typed_vals[name]) for name in type_names]
        positive_rates = [
            100.0 * sum(1 for value in typed_vals[name] if value > 0) / len(typed_vals[name])
            for name in type_names
        ]
        axes[0].barh(y, means, color="#4f8a63")
        axes[0].axvline(0.0, color="#666666", linewidth=1.0)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(type_names)
        axes[0].set_xlabel("mean val gain (percentage points)")
        axes[0].set_title("Mean validation gain after 1 generation")
        axes[0].grid(True, axis="x", alpha=0.25)
        axes[1].barh(y, positive_rates, color="#3568a8")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(type_names)
        axes[1].set_xlabel("positive actions (%)")
        axes[1].set_xlim(0, 105)
        axes[1].set_title("Share with validation gain > 0")
        axes[1].grid(True, axis="x", alpha=0.25)
        figure.suptitle(
            "Which action types help after one recovery generation (all starters)",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        save(figure, "005-action-type-val-gain-and-positive-rate.png")

    history_algs = [alg_id for alg_id in ALG_ORDER if alg_subset(all_completed, alg_id)]
    n_algs = len(history_algs)
    n_cols = 4
    n_rows = int(np.ceil(n_algs / n_cols))
    handles = [
        Line2D([0], [0], color="#3568a8", label="big"),
        Line2D([0], [0], color="#4f8a63", label="medium"),
    ]

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for index, alg_id in enumerate(history_algs):
        axis = axes_flat[index]
        for model_name, color, alpha in (
            ("big", "#3568a8", 0.55),
            ("medium_1conv_2linear", "#4f8a63", 0.55),
        ):
            subset = alg_subset(model_runs(model_name), alg_id)
            for run in subset:
                axis.plot([100.0 * float(v) for v in run["train_acc"]], color=color, alpha=alpha, linewidth=1.0)
        axis.set_title(_short(alg_id), fontsize=9)
        axis.grid(True, alpha=0.25)
        if index % n_cols == 0:
            axis.set_ylabel("train acc (%)")
        if index >= n_algs - n_cols:
            axis.set_xlabel("epoch")
    for index in range(n_algs, len(axes_flat)):
        axes_flat[index].axis("off")
    figure.legend(handles=handles, loc="upper right", fontsize=8)
    figure.suptitle(
        "Training histories for every simulation algorithm\nblue = big starter · green = medium starter",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save(figure, "005-training-curves-all-algorithms.png")

    def write_mean_fit_grid(
        metric_key: str,
        filename: str,
        ylabel: str,
        title: str,
    ) -> None:
        figure, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, sharey=True
        )
        axes_flat = np.atleast_1d(axes).ravel()
        for index, alg_id in enumerate(history_algs):
            axis = axes_flat[index]
            for model_name, color in (
                ("big", "#3568a8"),
                ("medium_1conv_2linear", "#4f8a63"),
            ):
                subset = alg_subset(model_runs(model_name), alg_id)
                series = [
                    [100.0 * float(v) for v in list(run[metric_key])]
                    for run in subset
                    if list(run.get(metric_key, []))
                ]
                if not series:
                    continue
                max_len = max(len(row) for row in series)
                means: list[float] = []
                stds: list[float] = []
                for epoch in range(max_len):
                    values = [row[epoch] for row in series if epoch < len(row)]
                    means.append(_mean(values))
                    stds.append(float(np.std(values)) if len(values) > 1 else 0.0)
                xs = list(range(len(means)))
                axis.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color,
                    alpha=0.18,
                )
                axis.plot(xs, means, color=color, linewidth=1.6, alpha=0.95)
            axis.set_title(_short(alg_id), fontsize=9)
            axis.grid(True, alpha=0.25)
            if index % n_cols == 0:
                axis.set_ylabel(ylabel)
            if index >= n_algs - n_cols:
                axis.set_xlabel("epoch")
        for index in range(n_algs, len(axes_flat)):
            axes_flat[index].axis("off")
        figure.legend(handles=handles, loc="upper right", fontsize=8)
        figure.suptitle(title, fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        save(figure, filename)

    write_mean_fit_grid(
        "train_acc",
        "005-training-curves-mean-std-best-fit.png",
        "train acc (%)",
        "Training mean ± std\n"
        "solid = mean · band = ±1 std · blue = big · green = medium",
    )

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for index, alg_id in enumerate(history_algs):
        axis = axes_flat[index]
        for model_name, color, alpha in (
            ("big", "#3568a8", 0.55),
            ("medium_1conv_2linear", "#4f8a63", 0.55),
        ):
            subset = alg_subset(model_runs(model_name), alg_id)
            for run in subset:
                axis.plot([100.0 * float(v) for v in run["val_acc"]], color=color, alpha=alpha, linewidth=1.0)
        axis.set_title(_short(alg_id), fontsize=9)
        axis.grid(True, alpha=0.25)
        if index % n_cols == 0:
            axis.set_ylabel("val acc (%)")
        if index >= n_algs - n_cols:
            axis.set_xlabel("epoch")
    for index in range(n_algs, len(axes_flat)):
        axes_flat[index].axis("off")
    figure.legend(handles=handles, loc="upper right", fontsize=8)
    figure.suptitle(
        "Validation histories for every simulation algorithm\nblue = big starter · green = medium starter",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save(figure, "005-validation-curves-all-algorithms.png")

    write_mean_fit_grid(
        "val_acc",
        "005-validation-curves-mean-std-best-fit.png",
        "val acc (%)",
        "Validation mean ± std\n"
        "solid = mean · band = ±1 std · blue = big · green = medium",
    )

    depth1_runs = [run for run in all_completed if str(run["alg_id"]) in DEPTH1_ALGS]
    lookahead_runs = [run for run in all_completed if str(run["alg_id"]) in LOOKAHEAD_ALGS]
    group_rows = [
        (
            "depth-1 only\n(no look-ahead)",
            depth1_runs,
            "#8a6d3b",
        ),
        (
            "look-ahead /\nhybrid",
            lookahead_runs,
            "#5c6b8a",
        ),
    ]
    if any(row[1] for row in group_rows):
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        xs = list(range(len(group_rows)))
        train_means = [
            100.0 * _mean([float(run["final_train_acc"]) for run in row[1]]) if row[1] else 0.0
            for row in group_rows
        ]
        val_means = [
            100.0 * _mean([float(run["final_val_acc"]) for run in row[1]]) if row[1] else 0.0
            for row in group_rows
        ]
        axis.bar([x - 0.18 for x in xs], train_means, width=0.32, label="mean final train", color="#555555")
        axis.bar([x + 0.18 for x in xs], val_means, width=0.32, label="mean final val", color="#7a6bb5")
        for index, (_label, subset, _color) in enumerate(group_rows):
            axis.text(
                index,
                max(train_means[index], val_means[index]),
                f"n={len(subset)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(xs)
        axis.set_xticklabels([row[0] for row in group_rows])
        axis.set_ylabel("accuracy (%)")
        axis.set_title(
            "Curiosity check: mean final accuracy by search group\n"
            "depth-1 = random/greedy/seq.halving/UGapE/succ.rejects · look-ahead = the rest"
        )
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)
        figure.tight_layout()
        save(figure, "005-lookahead-vs-depth1-final-accuracy.png")

    return list(written)


if __name__ == "__main__":
    paths = generate_charts()
    print(f"Wrote {len(paths)} chart(s)")
    for path in paths:
        print(path)
