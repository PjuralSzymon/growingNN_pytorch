"""Build Experiment 014 charts and a data snapshot from the saved run files.

Part A: same-checkpoint tests (paired/paired_results.json).
Part B: growing runs (growing/<step>/seed_<n>/history.pt and board/simulations/*.json).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/output/train_cifar10/runs/exp014_duplicate_action_design"
ASSETS = ROOT / "documentation/website/app/public/assets/experiments"
SNAPSHOT = ROOT / "documentation/website/data/experiments/experiment-014-duplicate-action-design.json"
SEEDS = (1, 2)
STEPS = [
    ("off", "0_off"),
    ("current", "1_current"),
    ("+side", "2_side"),
    ("+after_relu", "3_after_relu"),
    ("+scale (1.0)", "4_scale_1"),
    ("+merge", "5_merge_scale_1"),
]
STEP_COLORS = {
    "off": "#8a8984",
    "current": "#2a78d6",
    "+side": "#eb6834",
    "+after_relu": "#1baf7a",
    "+scale (1.0)": "#eda100",
    "+merge": "#e87ba4",
}
SEED_STYLE = {1: {"linestyle": "-", "marker": "o"}, 2: {"linestyle": "--", "marker": "s"}}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
BLOCK_COLORS = {"duplicate": "#1baf7a", "remove": "#e34948", "merge": "#6250d6"}

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.size": 9.5,
    "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    fig.savefig(ASSETS / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _block_op(text: str) -> str | None:
    if "Residual Block" not in text:
        return None
    for op in ("duplicate", "merge", "remove"):
        if f"'{op}'" in text:
            return op
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_paired() -> dict:
    return json.loads((RUNS / "paired/paired_results.json").read_text(encoding="utf-8"))


def load_run(folder: str, seed: int) -> dict:
    run_dir = RUNS / "growing" / folder / f"seed_{seed}"
    h = torch.load(run_dir / "history.pt", map_location="cpu", weights_only=False)
    actions = [
        {
            "generation": int(g),
            "name": str(n),
            "op": str(f) if n == "ResidualBlockAction" else None,
            "val_before": float(b),
            "val_after": float(a),
        }
        for g, n, f, b, a in zip(h["action_generation"], h["action_name"], h["action_from"],
                                 h["action_val_acc_before"], h["action_val_acc_after"])
    ]
    searches = []
    for path in sorted((run_dir / "board/simulations").glob("simulation_gen_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cands = [c for c in data.get("candidates") or [] if c.get("score") is not None]
        if not cands:
            continue
        scores = sorted((float(c["score"]) for c in cands), reverse=True)
        dropout = [float(c["score"]) for c in cands if "Dropout" in c.get("name", "")]
        dups = [float(c["score"]) for c in cands if _block_op(c.get("action", "")) == "duplicate"]
        chosen = next((c for c in cands if c.get("chosen")), None)
        searches.append({
            "generation": int(data.get("generation", 0)),
            "candidates": len(cands),
            "best": scores[0],
            "no_change": statistics.median(dropout) if dropout else None,
            "dropout_sd": statistics.pstdev(dropout) if len(dropout) > 1 else None,
            "best_duplicate": max(dups) if dups else None,
            "duplicate_rank": scores.index(max(dups)) + 1 if dups else None,
            "chosen_op": _block_op(chosen.get("action", "")) if chosen else None,
        })
    return {
        "seed": seed,
        "train_acc": [float(v) for v in h["train_acc"]],
        "val_acc": [float(v) for v in h["val_acc"]],
        "test_acc": [float(v) for v in h["test_acc"]],
        "param_count": [int(v) for v in h["param_count"]],
        "actions": actions,
        "searches": searches,
    }


def load_runs() -> dict[str, list[dict]]:
    return {label: [load_run(folder, s) for s in SEEDS] for label, folder in STEPS}


# ---------------------------------------------------------------------------
# Part A charts
# ---------------------------------------------------------------------------

def _gain_by_seed(table: list[dict]) -> dict[str, list[float]]:
    base = table[0]["val_after_generation_by_seed"]
    return {
        row["setting"]: [100 * (v - b) for v, b in zip(row["val_after_generation_by_seed"], base)]
        for row in table[1:] if row["available"]
    }


def plot_paired_ladder(paired: dict) -> None:
    tables = paired["tables"]
    panels = [
        ("Plain checkpoint (one block)", tables["Grow ladder, plain checkpoint"]),
        ("Checkpoint with an AddResConv side input", tables["Grow ladder, checkpoint with AddResConv side input"]),
    ]
    order = ["current", "+side", "+after_relu", "+scale"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (title, table) in zip(axes, panels):
        gains = _gain_by_seed(table)
        for i, setting in enumerate(order):
            color = STEP_COLORS["+scale (1.0)" if setting == "+scale" else setting]
            if setting not in gains:
                ax.text(i, 0.25, "not\noffered", ha="center", va="bottom", color=MUTED, fontsize=8.5)
                continue
            values = gains[setting]
            mean = statistics.mean(values)
            ax.bar(i, mean, width=0.6, color=color, zorder=2)
            for seed, v in zip(SEEDS, values):
                ax.scatter(i + (-0.12 if seed == 1 else 0.12), v, s=22, color="white", edgecolor=INK,
                           linewidth=0.9, marker=SEED_STYLE[seed]["marker"], zorder=3)
            ax.text(i, max(mean, max(values)) + 0.25, f"{mean:+.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(len(order)), ["current", "+side", "+after_relu", "+scale 0.3"])
        ax.set_title(title, loc="left")
        ax.grid(axis="y", zorder=0)
    axes[0].set_ylabel("Val accuracy vs no change\nafter one generation (pp)")
    axes[0].scatter([], [], marker="o", color="white", edgecolor=INK, label="seed 1")
    axes[0].scatter([], [], marker="s", color="white", edgecolor=INK, label="seed 2")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "014-paired-ladder.png")


def plot_paired_trajectory(paired: dict) -> None:
    panels = [
        ("Grow ladder, plain checkpoint", "Grow: duplicate variants (plain checkpoint)",
         {"no change": "off", "current": "current", "+after_relu": "+after_relu", "+scale": "+scale (1.0)"}),
        ("Shrink, two-block checkpoint", "Shrink: remove vs merge (two-block checkpoint)",
         {"no change": "off", "remove": "remove", "merge": "merge"}),
    ]
    stages = ["checkpoint", "right after\naction", "after\n1 epoch", "after one\ngeneration (10 ep.)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, (key, title, wanted) in zip(axes, panels):
        for row in paired["tables"][key]:
            if row["setting"] not in wanted:
                continue
            label = wanted[row["setting"]]
            color = STEP_COLORS.get(label) or BLOCK_COLORS[label]
            if label == "+scale (1.0)":
                label = "+scale 0.3"
            ys = [row["checkpoint_val"], row["checkpoint_val"] + row["val_change_now"],
                  row["val_after_1_epoch"], row["val_after_generation"]]
            ys = [100 * y for y in ys]
            ax.plot(range(4), ys, color=color, linewidth=2, marker="o", markersize=5,
                    label="no change" if label == "off" else label)
            ax.text(3.08, ys[-1], f"{ys[-1]:.1f}", va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(4), stages)
        ax.set_xlim(-0.15, 3.45)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Validation accuracy (%)")
        ax.grid(axis="y")
        ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "014-paired-trajectory.png")


def plot_scale_sweep(paired: dict) -> None:
    table = paired["tables"]["Scale sweep (side + after_relu), plain checkpoint"]
    base = table[0]
    rows = sorted(table[1:], key=lambda r: float(r["setting"].split()[1]))
    scales = [float(r["setting"].split()[1]) for r in rows]
    now = [100 * r["val_change_now"] for r in rows]
    one = [100 * (r["val_after_1_epoch"] - base["val_after_1_epoch"]) for r in rows]
    gen = [100 * (r["val_after_generation"] - base["val_after_generation"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for ys, label, color in [(now, "right after the action", "#e34948"),
                             (one, "after 1 epoch (what the search sees)", "#eda100"),
                             (gen, "after one generation", "#1baf7a")]:
        ax.plot(scales, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.text(scales[-1] + 0.03, ys[-1], f"{ys[-1]:+.1f}", va="center", fontsize=8.5)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Residual scale ε on the copy's last layer (0 = exact no-op, 1 = full quasi-identity)")
    ax.set_ylabel("Val accuracy vs no change (pp)")
    ax.set_xlim(-0.05, 1.15)
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "014-scale-sweep.png")


# ---------------------------------------------------------------------------
# Part B charts
# ---------------------------------------------------------------------------

def plot_final_test(runs: dict) -> None:
    labels = [label for label, _ in STEPS]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for i, label in enumerate(labels):
        values = [100 * r["test_acc"][-1] for r in runs[label]]
        mean = statistics.mean(values)
        ax.bar(i, mean, width=0.6, color=STEP_COLORS[label], zorder=2)
        for r, v in zip(runs[label], values):
            ax.scatter(i + (-0.12 if r["seed"] == 1 else 0.12), v, s=26, color="white", edgecolor=INK,
                       linewidth=0.9, marker=SEED_STYLE[r["seed"]]["marker"], zorder=3)
        ax.text(i, 51.2, f"mean\n{mean:.1f}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(50, 86)
    ax.set_ylabel("Final test accuracy (%)")
    ax.grid(axis="y", zorder=0)
    ax.scatter([], [], marker="o", color="white", edgecolor=INK, label="seed 1")
    ax.scatter([], [], marker="s", color="white", edgecolor=INK, label="seed 2")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    _save(fig, "014-final-test.png")


def _block_events(run: dict) -> list[tuple[int, str]]:
    return [((a["generation"] + 1) * 10, a["op"]) for a in run["actions"] if a["op"]]


def plot_histories(runs: dict, key: str, filename: str, ylabel: str, scale: float = 100.0) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    off_mean = [statistics.mean(v) * scale for v in zip(*(r[key] for r in runs["off"]))]
    for ax, (label, _) in zip(axes.flat, STEPS):
        if label != "off":
            ax.plot(range(1, len(off_mean) + 1), off_mean, color="#c9c8c3", linewidth=1.6, label="off, mean of seeds")
        for r in runs[label]:
            ys = [v * scale for v in r[key]]
            ax.plot(range(1, len(ys) + 1), ys, color=STEP_COLORS[label], linewidth=1.3,
                    linestyle=SEED_STYLE[r["seed"]]["linestyle"], label=f"seed {r['seed']}")
            for k, (epoch, op) in enumerate(_block_events(r)):
                ax.axvline(epoch, color=BLOCK_COLORS[op], linewidth=1.1, alpha=0.8,
                           linestyle=SEED_STYLE[r["seed"]]["linestyle"])
                ax.text(epoch + 1.5, 0.04 + 0.08 * (k + (r["seed"] - 1) * 2), f"{op} s{r['seed']}",
                        transform=ax.get_xaxis_transform(), fontsize=7.5, color=BLOCK_COLORS[op])
        ax.set_title(label, loc="left", color=INK)
        ax.grid(axis="y")
    for ax in axes[1]:
        ax.set_xlabel("Epoch (10 per generation)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, 1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, filename)


def plot_selection(runs: dict) -> None:
    labels = [label for label, _ in STEPS[1:]]
    metrics = [
        ("searches", "#c9c8c3"),
        ("duplicate offered", "#2a78d6"),
        ("duplicate in top 5", "#eda100"),
        ("duplicate chosen", "#1baf7a"),
    ]
    counts = {m: [] for m, _ in metrics}
    for label in labels:
        searches = [s for r in runs[label] for s in r["searches"]]
        counts["searches"].append(len(searches))
        counts["duplicate offered"].append(sum(s["best_duplicate"] is not None for s in searches))
        counts["duplicate in top 5"].append(sum((s["duplicate_rank"] or 99) <= 5 for s in searches))
        counts["duplicate chosen"].append(sum(s["chosen_op"] == "duplicate" for s in searches))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.19
    for j, (metric, color) in enumerate(metrics):
        xs = [i + (j - 1.5) * (width + 0.01) for i in range(len(labels))]
        ax.bar(xs, counts[metric], width=width, color=color, label=metric, zorder=2)
        for x, v in zip(xs, counts[metric]):
            ax.text(x, v + 0.8, str(v), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("MCTS searches (2 seeds)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", ncol=4, fontsize=8.5)
    ax.set_ylim(0, 70)
    fig.tight_layout()
    _save(fig, "014-duplicate-selection.png")


def plot_score_gaps(runs: dict) -> None:
    labels = [label for label, _ in STEPS[1:]]
    vs_best, vs_none, noise = [], [], []
    for label in labels:
        s = [x for r in runs[label] for x in r["searches"] if x["best_duplicate"] is not None]
        vs_best.append([100 * (x["best_duplicate"] - x["best"]) for x in s])
        vs_none.append([100 * (x["best_duplicate"] - x["no_change"]) for x in s if x["no_change"] is not None])
        noise.extend(100 * x["dropout_sd"] for r in runs[label] for x in r["searches"] if x["dropout_sd"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, data, title in [(axes[0], vs_none, "Best duplicate minus no change (median dropout)"),
                            (axes[1], vs_best, "Best duplicate minus the winning candidate")]:
        parts = ax.boxplot(data, positions=range(len(labels)), widths=0.5, patch_artist=True,
                           showfliers=False, medianprops={"color": INK})
        for patch, label in zip(parts["boxes"], labels):
            patch.set_facecolor(STEP_COLORS[label])
            patch.set_alpha(0.55)
            patch.set_edgecolor(STEP_COLORS[label])
        for i, values in enumerate(data):
            ax.scatter([i + 0.18] * len(values), values, s=9, color=STEP_COLORS[labels[i]], alpha=0.7, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, fontsize=8.5)
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
    sd = statistics.mean(noise)
    axes[0].axhspan(-sd, sd, color="#c9c8c3", alpha=0.35, zorder=0, label=f"±1 SD of no-op scores ({sd:.1f} pp)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Simulation score difference (pp)")
    fig.tight_layout()
    _save(fig, "014-score-gaps.png")
    return sd


def plot_action_mix(runs: dict) -> None:
    groups = {
        "AddResConvLayer": ("residual conv", "#2a78d6"),
        "AddSeqConvLayer": ("seq conv", "#6250d6"),
        "AddResLinearLayer": ("residual linear", "#eb6834"),
        "AddSeqLinearLayer": ("seq linear", "#eda100"),
        "AddSeqDropoutLayer": ("dropout", "#c9c8c3"),
        "AddNeurons": ("add neurons", "#e87ba4"),
        "DelNeurons": ("delete neurons", "#008300"),
        "DelLayer": ("delete layer", "#52514e"),
        "ResidualBlockAction": ("residual block", "#e34948"),
    }
    labels = [label for label, _ in STEPS]
    counts = {name: [] for name in groups}
    for label in labels:
        names = [a["name"] for r in runs[label] for a in r["actions"]]
        for name in groups:
            counts[name].append(sum(n == name for n in names))
    fig, ax = plt.subplots(figsize=(9, 3.9))
    left = [0] * len(labels)
    for name, (pretty, color) in groups.items():
        if not any(counts[name]):
            continue
        ax.barh(range(len(labels)), counts[name], left=left, color=color, edgecolor="white", linewidth=1.5,
                label=pretty, height=0.62)
        for i, v in enumerate(counts[name]):
            if v >= 3:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if color not in ("#c9c8c3", "#eda100") else INK)
        left = [a + b for a, b in zip(left, counts[name])]
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Executed actions (2 seeds × 29 searches)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "014-action-mix.png")


def block_events(runs: dict) -> list[dict]:
    """Every executed residual-block action with validation accuracy right before and after it."""
    return [{"step": label, "seed": r["seed"], **a}
            for label, _ in STEPS for r in runs[label] for a in r["actions"] if a["op"]]


def plot_context(runs: dict) -> dict:
    """Place Exp014 next to Exp010 and Exp013 (final validation accuracy on CIFAR-10, mean of seeds)."""
    off200 = statistics.mean(r["val_acc"][199] for r in runs["off"])
    rows = [
        # Exp010 and Exp013 values are copied from their published reports.
        ("Exp010: 296-param starter\n128 sim examples, 200 epochs", 46.74, "#c9c8c3"),
        ("Exp010: 296-param starter\ntuned search, 200 epochs", 59.16, "#c9c8c3"),
        ("Exp014 off: 2,970-param residual\nstarter, at epoch 200", 100 * off200, "#8a8984"),
        ("Exp013 block actions off\nsame starter, 300 epochs", 72.98, "#8a8984"),
        ("Exp014 off\n300 epochs", 100 * statistics.mean(r["val_acc"][-1] for r in runs["off"]), "#8a8984"),
        ("Exp014 +after_relu\n300 epochs", 100 * statistics.mean(r["val_acc"][-1] for r in runs["+after_relu"]), "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for i, (label, value, color) in enumerate(rows):
        ax.barh(i, value, color=color, height=0.62, zorder=2)
        ax.text(value + 0.8, i, f"{value:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 90)
    ax.set_xlabel("CIFAR-10 validation accuracy at the stated epoch, mean of seeds (%)")
    ax.grid(axis="x", zorder=0)
    fig.tight_layout()
    _save(fig, "014-context.png")
    return {label.replace("\n", " "): value for label, value, _ in rows}


# ---------------------------------------------------------------------------

def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    paired = load_paired()
    runs = load_runs()
    plot_paired_ladder(paired)
    plot_paired_trajectory(paired)
    plot_scale_sweep(paired)
    plot_final_test(runs)
    plot_histories(runs, "val_acc", "014-val-histories.png", "Validation accuracy (%)")
    plot_histories(runs, "param_count", "014-param-histories.png", "Parameters", scale=1.0)
    plot_selection(runs)
    noise_sd = plot_score_gaps(runs)
    plot_action_mix(runs)
    events = block_events(runs)
    context = plot_context(runs)

    snapshot = {
        "experiment": "014",
        "source": str(RUNS.relative_to(ROOT)).replace("\\", "/"),
        "paired": paired,
        "noise_sd_pp": noise_sd,
        "context_final_val_pct": context,
        "block_action_events": events,
        "runs": {
            label: [{
                "seed": r["seed"],
                "final_train": r["train_acc"][-1],
                "final_val": r["val_acc"][-1],
                "final_test": r["test_acc"][-1],
                "final_params": r["param_count"][-1],
                "epochs": len(r["val_acc"]),
                "actions": r["actions"],
                "searches": r["searches"],
            } for r in runs[label]]
            for label, _ in STEPS
        },
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    for label, _ in STEPS:
        print(label, [f"{100 * r['test_acc'][-1]:.2f}" for r in runs[label]],
              [f"{100 * r['val_acc'][-1]:.2f}" for r in runs[label]],
              [r["param_count"][-1] for r in runs[label]],
              [(a["generation"], a["op"], round(a["val_before"], 3), round(a["val_after"], 3))
               for r in runs[label] for a in r["actions"] if a["op"]])
    print("noise sd pp", round(noise_sd, 2))


if __name__ == "__main__":
    main()
