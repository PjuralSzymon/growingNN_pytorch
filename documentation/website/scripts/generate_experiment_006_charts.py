"""Generate charts for Experiment 006: neuron-resize action ratio pairs.

Measured figures (boards or snapshot):
- final train/val accuracy by group with seed markers
- start vs final parameter counts by group
- executed action mix by group
- chosen simulation actions by group
- neuron-resize candidate presence vs scoring
- mean composite SimulationScore by action family
- training-accuracy curves colored by group
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
DEFAULT_RUNS = _RUNS_ROOT / "exp006_neuron_resize_actions"
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-006-neuron-resize-actions.json"
DEFAULT_SIM_ANALYSIS = SITE / "data" / "experiments" / "experiment-006-simulation-action-analysis.json"
_ALLOWED_SNAPSHOT_ROOT = (SITE / "data" / "experiments").resolve()
_ALLOWED_OUTPUT_ROOT = (SITE / "app" / "public" / "assets" / "experiments").resolve()
_ALLOWED_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()

GROUP_ORDER = ("none", "add11_del01", "add15_del05", "add20_del09")
GROUP_LABELS = {
    "none": "none",
    "add11_del01": "add1.1 / del0.1",
    "add15_del05": "add1.5 / del0.5",
    "add20_del09": "add2.0 / del0.9",
}
GROUP_COLORS = {
    "none": "#6b7280",
    "add11_del01": "#3568a8",
    "add15_del05": "#4f8a63",
    "add20_del09": "#d18b2c",
}
ACTION_SCORE_ORDER = (
    "Add Neurons Action",
    "Delete Neurons Action",
    "Add Res Conv Layer Action",
    "Add Res Linear Layer Action",
    "Add Seq Conv Layer Action",
    "Add Seq Linear Layer Action",
    "Add Seq Dropout Layer Action",
    "Delete Layer Action",
)


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
        main = json.loads(main_resolved.read_text(encoding="utf-8"))
        if main.get("status") != "completed":
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
    payload = {"experiment": "006", "folder": folder, "runs": runs}
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
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_xlabel("Neuron-resize group")
    ax.legend()
    ax.set_title("Final accuracy by neuron-resize group")
    fig.tight_layout()
    fig.savefig(output_dir / "006-final-accuracy-by-group.png", dpi=150)
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
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Parameter count")
    ax.set_xlabel("Neuron-resize group")
    ax.legend()
    ax.set_title("Parameter growth by neuron-resize group")
    fig.tight_layout()
    fig.savefig(output_dir / "006-param-growth-by-group.png", dpi=150)
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
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Mean executed actions")
    ax.set_xlabel("Neuron-resize group")
    ax.legend(fontsize=8, ncols=2)
    ax.set_title("Executed action mix by neuron-resize group")
    fig.tight_layout()
    fig.savefig(output_dir / "006-action-composition-by-group.png", dpi=150)
    plt.close(fig)


def _neuron_kind(action: str | None) -> str | None:
    if not action:
        return None
    if "Add Neurons" in action:
        return "AddNeurons"
    if "Delete Neurons" in action:
        return "DelNeurons"
    return None


def _short_action(action: str | None) -> str:
    if not action:
        return "(none)"
    match = re.search(r"\(\s*([^:]+?)Action", action)
    if match:
        return match.group(1).strip() + " Action"
    return str(action)[:60]


def summarize_action_scores(
    scores_by_label: dict[str, list[float]],
) -> dict[str, dict[str, float | int | None]]:
    """Return n, mean, min, and max composite score for each action label."""
    labels = [label for label in ACTION_SCORE_ORDER if label in scores_by_label]
    extra = sorted(label for label in scores_by_label if label not in ACTION_SCORE_ORDER)
    summary: dict[str, dict[str, float | int | None]] = {}
    for label in (*labels, *extra):
        values = scores_by_label.get(label, [])
        summary[label] = {
            "n": len(values),
            "mean": _mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def build_simulation_action_analysis(runs_dir: Path) -> dict[str, object]:
    """Aggregate simulation candidate scores from completed boards."""
    rows: list[dict[str, object]] = []
    if runs_dir.exists():
        for sim_path in sorted(runs_dir.rglob("simulations/simulation_gen_*.json")):
            parts = sim_path.relative_to(runs_dir).parts
            if len(parts) < 3:
                continue
            group_id, seed = parts[0], parts[2]
            main_path = sim_path.parents[1] / "main.json"
            if not main_path.is_file():
                continue
            main = json.loads(main_path.read_text(encoding="utf-8"))
            if main.get("status") != "completed":
                continue
            sim = json.loads(sim_path.read_text(encoding="utf-8"))
            candidates = list(sim.get("candidates") or [])
            scored = [c for c in candidates if c.get("score") is not None]
            ranked = sorted(scored, key=lambda c: float(c["score"]), reverse=True)
            best_score = float(ranked[0]["score"]) if ranked else None
            rank_map = {id(c): i + 1 for i, c in enumerate(ranked)}
            chosen = next((c for c in candidates if c.get("chosen")), None)
            chosen_score = (
                chosen.get("score")
                if chosen and chosen.get("score") is not None
                else sim.get("scoreChosen")
            )
            neuron_entries = []
            scored_by_label: dict[str, list[float]] = defaultdict(list)
            for cand in candidates:
                label = _short_action(str(cand.get("action") or ""))
                score = cand.get("score")
                if score is not None:
                    scored_by_label[label].append(float(score))
                kind = _neuron_kind(str(cand.get("action") or ""))
                if not kind:
                    continue
                neuron_entries.append(
                    {
                        "kind": kind,
                        "score": float(score) if score is not None else None,
                        "scored": score is not None,
                        "chosen": bool(cand.get("chosen")),
                        "rank": rank_map.get(id(cand)) if score is not None else None,
                        "gap_to_best": (
                            best_score - float(score)
                            if score is not None and best_score is not None
                            else None
                        ),
                    }
                )
            rows.append(
                {
                    "group": group_id,
                    "seed": seed,
                    "generation": sim.get("generation"),
                    "chosen_label": _short_action(
                        chosen.get("action") if chosen else sim.get("actionChosen")
                    ),
                    "chosen_score": float(chosen_score) if chosen_score is not None else None,
                    "neuron_entries": neuron_entries,
                    "scored_by_label": dict(scored_by_label),
                }
            )

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)

    groups: dict[str, object] = {}
    for group_id in GROUP_ORDER:
        group_rows = by_group.get(group_id, [])
        neuron_all = [e for r in group_rows for e in r["neuron_entries"]]
        neuron_scored = [e for e in neuron_all if e["scored"]]
        gaps = [e["gap_to_best"] for e in neuron_scored if e["gap_to_best"] is not None]
        ranks = [e["rank"] for e in neuron_scored if e["rank"] is not None]
        near = sorted(
            (
                {
                    "seed": r["seed"],
                    "generation": r["generation"],
                    **e,
                    "winner": r["chosen_label"],
                    "winner_score": r["chosen_score"],
                }
                for r in group_rows
                for e in r["neuron_entries"]
                if e["scored"]
            ),
            key=lambda item: item["gap_to_best"] if item["gap_to_best"] is not None else 9e9,
        )
        group_scores: dict[str, list[float]] = defaultdict(list)
        for row in group_rows:
            for label, values in dict(row.get("scored_by_label") or {}).items():
                group_scores[str(label)].extend(float(v) for v in values)
        groups[group_id] = {
            "sims": len(group_rows),
            "chosen_mix": dict(Counter(str(r["chosen_label"]) for r in group_rows)),
            "sims_with_neuron_in_pool": sum(1 for r in group_rows if r["neuron_entries"]),
            "sims_with_neuron_scored": sum(
                1 for r in group_rows if any(e["scored"] for e in r["neuron_entries"])
            ),
            "neuron_pool_count": len(neuron_all),
            "neuron_scored_count": len(neuron_scored),
            "neuron_unscored_count": len(neuron_all) - len(neuron_scored),
            "neuron_chosen": sum(1 for e in neuron_all if e["chosen"]),
            "add_pool": sum(1 for e in neuron_all if e["kind"] == "AddNeurons"),
            "del_pool": sum(1 for e in neuron_all if e["kind"] == "DelNeurons"),
            "mean_neuron_rank": _mean([float(v) for v in ranks]) if ranks else None,
            "mean_gap_to_best": _mean([float(v) for v in gaps]) if gaps else None,
            "min_gap_to_best": min(gaps) if gaps else None,
            "near_misses": near[:10],
            "mean_score_by_action": summarize_action_scores(group_scores),
        }
    overall_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for label, values in dict(row.get("scored_by_label") or {}).items():
            overall_scores[str(label)].extend(float(v) for v in values)
    return {
        "n_simulations": len(rows),
        "overall_chosen_mix": dict(Counter(str(r["chosen_label"]) for r in rows)),
        "overall_mean_score_by_action": summarize_action_scores(overall_scores),
        "groups": groups,
    }


def plot_simulation_chosen_actions(analysis: dict[str, object], output_dir: Path) -> None:
    groups = analysis["groups"]
    actions = sorted(
        {
            label
            for group_id in GROUP_ORDER
            for label in groups[group_id]["chosen_mix"]
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(GROUP_ORDER))
    bottoms = np.zeros(len(GROUP_ORDER))
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(actions):
        heights = [groups[g]["chosen_mix"].get(label, 0) for g in GROUP_ORDER]
        ax.bar(x, heights, bottom=bottoms, label=label.replace(" Action", ""), color=cmap(idx % 10))
        bottoms = bottoms + np.array(heights, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Executed simulations")
    ax.set_xlabel("Neuron-resize group")
    ax.legend(fontsize=8, ncols=2)
    ax.set_title("Chosen simulation actions by group")
    fig.tight_layout()
    fig.savefig(output_dir / "006-simulation-chosen-actions-by-group.png", dpi=150)
    plt.close(fig)


def plot_neuron_candidate_scoring(analysis: dict[str, object], output_dir: Path) -> None:
    groups = analysis["groups"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.35
    pool = [groups[g]["sims_with_neuron_in_pool"] for g in GROUP_ORDER]
    scored = [groups[g]["sims_with_neuron_scored"] for g in GROUP_ORDER]
    sims = [groups[g]["sims"] for g in GROUP_ORDER]
    ax.bar(x - width / 2, pool, width, label="neuron in candidate pool", color="#3568a8")
    ax.bar(x + width / 2, scored, width, label="neuron scored at least once", color="#d18b2c")
    for i, count in enumerate(sims):
        ax.text(i, max(count, 1) + 0.2, f"n_sim={count}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Simulations")
    ax.set_xlabel("Neuron-resize group")
    ax.set_title("Neuron-resize presence vs scoring in simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "006-neuron-candidate-scoring.png", dpi=150)
    plt.close(fig)


def plot_mean_simulation_scores(analysis: dict[str, object], output_dir: Path) -> None:
    groups = analysis["groups"]
    labels = [
        label
        for label in ACTION_SCORE_ORDER
        if any(
            int((groups[g].get("mean_score_by_action") or {}).get(label, {}).get("n") or 0) > 0
            for g in GROUP_ORDER
        )
    ]
    extra = sorted(
        {
            label
            for g in GROUP_ORDER
            for label in (groups[g].get("mean_score_by_action") or {})
            if label not in ACTION_SCORE_ORDER
            and int((groups[g]["mean_score_by_action"][label].get("n") or 0)) > 0
        }
    )
    labels.extend(extra)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    width = 0.18
    offsets = (np.arange(len(GROUP_ORDER)) - (len(GROUP_ORDER) - 1) / 2) * width
    for i, group_id in enumerate(GROUP_ORDER):
        stats = groups[group_id].get("mean_score_by_action") or {}
        xs: list[float] = []
        ys: list[float] = []
        for j, label in enumerate(labels):
            item = stats.get(label) or {}
            mean = item.get("mean")
            if not item.get("n") or mean is None:
                continue
            xs.append(float(x[j] + offsets[i]))
            ys.append(float(mean))
        ax.bar(
            xs,
            ys,
            width,
            label=GROUP_LABELS[group_id],
            color=GROUP_COLORS[group_id],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label.replace(" Action", "").replace(" Layer", "") for label in labels],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Mean composite SimulationScore")
    ax.set_xlabel("Scored root action")
    ax.set_title("Mean simulation score by action")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "006-mean-simulation-score-by-action.png", dpi=150)
    plt.close(fig)


def plot_training_curves(runs: list[dict[str, object]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in runs:
        group_id = str(run["group_id"])
        ys = [v * 100 for v in run["train_acc"]]
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
    ax.set_ylabel("Training accuracy (%)")
    ax.set_title("Training accuracy curves by neuron-resize group")
    fig.tight_layout()
    fig.savefig(output_dir / "006-training-curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    runs_dir = DEFAULT_RUNS
    output_dir = _resolve_under_allowed_root(DEFAULT_OUTPUT, _ALLOWED_OUTPUT_ROOT)
    snapshot_path = DEFAULT_SNAPSHOT
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs_or_snapshot(runs_dir, snapshot_path)
    if not runs:
        print(
            f"No completed Exp 006 runs under {runs_dir}. "
            f"Snapshot empty or missing: {snapshot_path}"
        )
        return
    plot_final_accuracy(runs, output_dir)
    plot_param_growth(runs, output_dir)
    plot_action_composition(runs, output_dir)
    sim_analysis = build_simulation_action_analysis(runs_dir)
    sim_snapshot = _resolve_under_allowed_root(DEFAULT_SIM_ANALYSIS, _ALLOWED_SNAPSHOT_ROOT)
    if sim_analysis["n_simulations"]:
        sim_snapshot.parent.mkdir(parents=True, exist_ok=True)
        sim_snapshot.write_text(json.dumps(sim_analysis, indent=2), encoding="utf-8")
    elif sim_snapshot.exists():
        sim_analysis = json.loads(sim_snapshot.read_text(encoding="utf-8"))
    plot_simulation_chosen_actions(sim_analysis, output_dir)
    plot_neuron_candidate_scoring(sim_analysis, output_dir)
    plot_mean_simulation_scores(sim_analysis, output_dir)
    plot_training_curves(runs, output_dir)
    print(
        f"Wrote Exp 006 charts for {len(runs)} completed runs "
        f"({sim_analysis['n_simulations']} simulations) to {output_dir}"
    )


if __name__ == "__main__":
    main()
