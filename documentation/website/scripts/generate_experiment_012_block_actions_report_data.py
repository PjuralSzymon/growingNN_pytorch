"""Regenerate Experiment 012's JSON snapshot and figures from history.pt files."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
# Completed runs stayed in the exp010 folder when this study was renumbered to 011.
RUNS = ROOT / "experiments/output/train_cifar10/runs/exp010_residual_block_actions"
SNAPSHOT = ROOT / "documentation/website/data/experiments/experiment-012-residual-block-actions.json"
ASSETS = ROOT / "documentation/website/app/public/assets/experiments"
DATASETS = ("mnist", "cifar10", "cifar100")
ARMS = ("block_actions_on", "block_actions_off")
ARM_LABELS = {"block_actions_on": "Block actions enabled", "block_actions_off": "Block actions disabled"}
COLORS = {"block_actions_on": "#2563eb", "block_actions_off": "#f97316"}


def load_runs() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            for path in sorted((RUNS / dataset / arm).glob("seed_*/history.pt")):
                history = torch.load(path, map_location="cpu", weights_only=False)
                names = list(history.get("action_name", []))
                actions = []
                for i, name in enumerate(names):
                    actions.append({
                        "generation": int(history["action_generation"][i]),
                        "name": str(name),
                        "from": str(history.get("action_from", [""] * len(names))[i]),
                        "to": str(history.get("action_to", [""] * len(names))[i]),
                        "val_before": float(history["action_val_acc_before"][i]),
                        "val_after": float(history["action_val_acc_after"][i]),
                    })
                rows.append({
                    "dataset": dataset,
                    "arm": arm,
                    "seed": int(history["seed"]),
                    "written": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    "train": [float(x) for x in history.get("train_acc", [])],
                    "val": [float(x) for x in history.get("val_acc", [])],
                    "test": [float(x) for x in history.get("test_acc", [])],
                    "params": [int(x) for x in history.get("param_count", [])],
                    "actions": actions,
                    "action_counts": dict(Counter(names)),
                })
    return rows


def plot_histories(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        for arm in ARMS:
            group = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            for i, run in enumerate(group):
                if not run["train"]:
                    continue
                x = range(1, len(run["train"]) + 1)
                alpha = 0.55 if len(group) > 1 else 0.9
                ax.plot(x, run["train"], color=COLORS[arm], alpha=alpha,
                        label=f"{ARM_LABELS[arm]} train" if i == 0 else None)
                ax.plot(x, run["val"], color=COLORS[arm], alpha=alpha, linestyle="--",
                        label=f"{ARM_LABELS[arm]} val" if i == 0 else None)
        ax.set_title({"mnist": "MNIST", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}[dataset])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(left=1)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Experiment 012: training and validation histories (each line is one seed)")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(ASSETS / "012-training-histories.png", dpi=170)
    plt.close(fig)


def plot_test(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, dataset in zip(axes, DATASETS):
        for x, arm in enumerate(ARMS):
            group = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm and r["test"]]
            vals = [run["test"][-1] for run in group]
            if vals:
                ax.scatter([x + (i - (len(vals)-1)/2)*0.08 for i in range(len(vals))], vals,
                           color=COLORS[arm], zorder=3)
                ax.plot([x-0.16, x+0.16], [sum(vals)/len(vals)]*2, color=COLORS[arm], lw=3)
        ax.set_xticks([0, 1], ["Enabled", "Disabled"])
        ax.set_title(dataset.upper() if dataset != "cifar10" else "CIFAR-10")
        ax.set_ylabel("Final test accuracy")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Final official-test accuracy by seed; horizontal bars show arm means")
    fig.tight_layout()
    fig.savefig(ASSETS / "012-final-test-accuracy.png", dpi=170)
    plt.close(fig)


def plot_actions(rows: list[dict]) -> None:
    totals: dict[tuple[str, str], Counter] = {}
    action_types = sorted({name for row in rows for name in row["action_counts"]})
    for dataset in DATASETS:
        for arm in ARMS:
            counter = Counter()
            for row in rows:
                if row["dataset"] == dataset and row["arm"] == arm:
                    counter.update(row["action_counts"])
            totals[(dataset, arm)] = counter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    bottom_by_ax = []
    for ax, dataset in zip(axes, DATASETS):
        bottoms = [0, 0]
        for action in action_types:
            vals = [totals[(dataset, arm)].get(action, 0) for arm in ARMS]
            ax.bar([0, 1], vals, bottom=bottoms, label=action)
            bottoms = [bottoms[i] + vals[i] for i in range(2)]
        ax.set_xticks([0, 1], ["Enabled", "Disabled"])
        ax.set_title(dataset.upper() if dataset != "cifar10" else "CIFAR-10")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Executed actions summed across completed seeds")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Chosen actions by dataset and arm (all completed seeds)")
    fig.tight_layout(rect=(0, 0.19, 1, 0.92))
    fig.savefig(ASSETS / "012-action-counts.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    outcome_groups: dict[str, list[float]] = {}
    for row in rows:
        for action in row["actions"]:
            outcome_groups.setdefault(action["name"], []).append(
                100.0 * (action["val_after"] - action["val_before"])
            )
    action_outcomes = {
        name: {
            "count": len(deltas),
            "mean_immediate_validation_change_percentage_points": sum(deltas) / len(deltas),
            "improved": sum(delta > 1e-9 for delta in deltas),
            "declined": sum(delta < -1e-9 for delta in deltas),
            "unchanged": sum(abs(delta) <= 1e-9 for delta in deltas),
        }
        for name, deltas in sorted(outcome_groups.items())
    }
    result = {
        "experiment": "012-residual-block-actions",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(RUNS.relative_to(ROOT)),
        "datasets": list(DATASETS),
        "arms": ARMS,
        "expected_runs": 12,
        "completed_runs": len(rows),
        "runs": rows,
        "selected_action_outcomes": action_outcomes,
        "score_data_note": (
            "Histories store immediate validation accuracy before and after the selected action. "
            "They do not store per-candidate simulation scores or generated-action counts."
        ),
    }
    SNAPSHOT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_histories(rows)
    plot_test(rows)
    plot_actions(rows)
    print(f"Loaded {len(rows)} completed histories; wrote {SNAPSHOT}")
    print(f"Wrote charts under {ASSETS}")


if __name__ == "__main__":
    main()
