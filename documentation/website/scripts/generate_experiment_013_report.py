"""Build interim Experiment 013 charts and a data snapshot from saved run files."""

from __future__ import annotations

import json
import pickle
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
# Completed and partial runs stayed in the exp012 folder when this study was renumbered to 013.
RUNS = ROOT / "experiments/output/train_cifar10/runs/exp012_residual_block_multidataset"
ASSETS = ROOT / "documentation/website/app/public/assets/experiments"
SNAPSHOT = ROOT / "documentation/website/data/experiments/experiment-013-residual-block-multidataset.json"
DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
ARMS = ("block_actions_on", "block_actions_off")
SEEDS = (1, 2, 3)
COLORS = {1: "#3568a8", 2: "#e4572e", 3: "#2a9d68"}
ARM_LABELS = {"block_actions_on": "Enabled", "block_actions_off": "Disabled"}
DATASET_LABELS = {"mnist": "MNIST", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny_imagenet": "Tiny ImageNet"}


def _load_pickle_history(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return pickle.loads(archive.read("history/data.pkl"))


def load_records() -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    candidates: list[dict] = []
    for dataset in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                run_dir = RUNS / dataset / arm / f"seed_{seed}"
                main_path = run_dir / "board/main.json"
                if not main_path.is_file():
                    records.append({"dataset": dataset, "arm": arm, "seed": seed, "status": "not_started", "epochs": []})
                    continue
                main = json.loads(main_path.read_text(encoding="utf-8"))
                metrics_path = run_dir / "board/metrics/training.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {"epochs": []}
                history_path = run_dir / "history.pt"
                history = _load_pickle_history(history_path) if history_path.is_file() else None
                epochs = metrics.get("epochs", [])
                record = {
                    "dataset": dataset,
                    "arm": arm,
                    "seed": seed,
                    "status": main.get("status", "unknown"),
                    "epochs": epochs,
                    "completed_epochs": len(epochs),
                    "main": main,
                    "history": history,
                    "final_train_accuracy": float(epochs[-1]["trainAcc"]) if epochs else None,
                    "final_validation_accuracy": float(epochs[-1]["valAcc"]) if epochs else None,
                    "final_test_accuracy": float(history["test_acc"][-1]) if history and history.get("test_acc") else None,
                    "final_parameters": int(epochs[-1]["paramCount"]) if epochs else None,
                    "stopped_by_validation_target": bool(history.get("stopped_by_validation_target")) if history else None,
                    "started": main.get("experimentStartedOn"),
                    "updated": main.get("lastUpdate"),
                    "elapsed_seconds": main.get("trainingTimeElapsedSec"),
                }
                records.append(record)

                if history:
                    record["actions"] = [
                        {"generation": int(generation), "name": str(name)}
                        for generation, name in zip(history.get("action_generation", []), history.get("action_name", []))
                    ]
                else:
                    record["actions"] = [
                        {"generation": int(row["generation"]), "name": str(row["actionExecuted"].get("action", ""))}
                        for row in main.get("generationTimeline", []) if row.get("actionExecuted")
                    ]

                simulation_dir = run_dir / "board/simulations"
                if simulation_dir.is_dir():
                    for simulation_path in simulation_dir.glob("simulation_gen_*.json"):
                        payload = json.loads(simulation_path.read_text(encoding="utf-8"))
                        chosen = str(payload.get("actionChosen") or "")
                        for row in payload.get("candidates", []):
                            action = str(row.get("action") or "")
                            if "Residual Block Action" not in action or row.get("score") is None:
                                continue
                            kind = "duplicate" if "duplicate" in action.lower() else "remove"
                            candidates.append({
                                "dataset": dataset,
                                "arm": arm,
                                "seed": seed,
                                "generation": int(payload.get("generation", 0)),
                                "kind": kind,
                                "score": float(row["score"]),
                                "selected": action.strip() == chosen.strip(),
                            })
    return records, candidates


def _plot_training(records: list[dict]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharey="row")
    for row, dataset in enumerate(DATASETS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            runs = [r for r in records if r["dataset"] == dataset and r["arm"] == arm and r["epochs"]]
            for record in sorted(runs, key=lambda r: r["seed"]):
                epochs = record["epochs"]
                x = [int(e["globalEpoch"]) + 1 for e in epochs]
                color = COLORS[record["seed"]]
                suffix = " (partial)" if record["status"] != "completed" else ""
                label = f"Seed {record['seed']}{suffix}"
                ax.plot(x, [100 * e["trainAcc"] for e in epochs], color=color, linewidth=1.35, label=f"{label}: train")
                ax.plot(x, [100 * e["valAcc"] for e in epochs], color=color, linestyle="--", linewidth=1.25, label=f"{label}: val")
                if record["status"] != "completed":
                    ax.scatter([x[-1]], [100 * epochs[-1]["valAcc"]], marker="x", s=38, color=color, zorder=5)
            if not runs:
                ax.text(0.5, 0.5, "No runs started", transform=ax.transAxes, ha="center", va="center", color="#666666")
            ax.axhline(90, color="#8b5cf6", linestyle=":", linewidth=1.0, label="90% validation stop" if row == 0 and col == 0 else None)
            ax.set_title(f"{DATASET_LABELS[dataset]} · {ARM_LABELS[arm]}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)
            max_epoch = max((r["completed_epochs"] for r in runs), default=10)
            ax.set_xlim(1, max(20, ((max_epoch + 9) // 10) * 10))
            ax.grid(alpha=0.22)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Experiment 013: training and validation accuracy", fontsize=15)
    fig.text(0.5, 0.006, "Solid=train, dashed=validation. Colors identify seeds; × marks the latest point of an incomplete run. A run stops at 90% validation or 300 epochs.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(ASSETS / "013-training-histories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_parameters(records: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, dataset in zip(axes.flat, DATASETS):
        for record in sorted((r for r in records if r["dataset"] == dataset and r["epochs"]), key=lambda r: (r["arm"], r["seed"])):
            x = [int(e["globalEpoch"]) + 1 for e in record["epochs"]]
            y = [int(e["paramCount"]) for e in record["epochs"]]
            linestyle = "-" if record["arm"] == "block_actions_on" else "--"
            suffix = " (partial)" if record["status"] != "completed" else ""
            ax.plot(x, y, color=COLORS[record["seed"]], linestyle=linestyle, linewidth=1.5,
                    label=f"Seed {record['seed']} · {ARM_LABELS[record['arm']]}{suffix}")
            if record["status"] != "completed":
                ax.scatter([x[-1]], [y[-1]], marker="x", s=38, color=COLORS[record["seed"]], zorder=5)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Parameters")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Experiment 013: parameter count during training", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ASSETS / "013-parameter-histories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_test_accuracy(records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    arm_colors = {"block_actions_on": "#7c3aed", "block_actions_off": "#64748b"}
    for ax, dataset in zip(axes, DATASETS):
        for x, arm in enumerate(ARMS):
            rows = [r for r in records if r["dataset"] == dataset and r["arm"] == arm and r["status"] == "completed" and r["final_test_accuracy"] is not None]
            ys = [100 * r["final_test_accuracy"] for r in rows]
            xs = [x + (i - (len(ys) - 1) / 2) * 0.08 for i in range(len(ys))]
            ax.scatter(xs, ys, color=arm_colors[arm], edgecolor="#333333", linewidth=0.4, s=36, zorder=3)
            if ys:
                mean = statistics.mean(ys)
                ax.hlines(mean, x - 0.28, x + 0.28, color=arm_colors[arm], linewidth=2)
                ax.text(x, mean + 1.2, f"mean {mean:.1f}% · n={len(ys)}", ha="center", fontsize=7, color=arm_colors[arm])
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(range(2), ["Enabled", "Disabled"])
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 013: final official test accuracy for completed runs")
    fig.text(0.5, 0.01, "Dots are individual seeds; horizontal marks are means. Tiny ImageNet currently has one completed enabled seed and no completed disabled control.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(ASSETS / "013-final-test-accuracy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate_scores(candidates: list[dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    colors = {"duplicate": "#7c3aed", "remove": "#94a3b8"}
    kinds = ("duplicate", "remove")
    for ax, dataset in zip(axes, DATASETS):
        rows = [c for c in candidates if c["dataset"] == dataset and c["arm"] == "block_actions_on"]
        for x, kind in enumerate(kinds):
            values = [c["score"] for c in rows if c["kind"] == kind]
            if values:
                jitter = [x + (i - (len(values) - 1) / 2) * min(0.035, 0.7 / max(len(values), 1)) for i in range(len(values))]
                ax.scatter(jitter, values, color=colors[kind], alpha=0.68, edgecolor="#444444", linewidth=0.3, s=24)
                mean = statistics.mean(values)
                ax.hlines(mean, x - 0.25, x + 0.25, color=colors[kind], linewidth=2)
                chosen = sum(c["selected"] for c in rows if c["kind"] == kind)
                ax.text(x, min(0.98, mean + 0.05), f"mean {mean:.2f}\nn={len(values)}, selected={chosen}", ha="center", fontsize=7)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(range(2), ["Duplicate", "Remove"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("MCTS candidate score (validation accuracy)")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 013: residual-block candidate scores in enabled arms")
    fig.text(0.5, 0.01, "Points are recorded candidate scores from simulation JSON. Tiny ImageNet includes the partial enabled seed 2 run.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(ASSETS / "013-block-candidate-scores.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def build_snapshot(records: list[dict], candidates: list[dict]) -> dict:
    groups = []
    for dataset in DATASETS:
        for arm in ARMS:
            rows = [r for r in records if r["dataset"] == dataset and r["arm"] == arm and r["status"] == "completed"]
            groups.append({
                "dataset": dataset,
                "arm": arm,
                "completedSeeds": [r["seed"] for r in rows],
                "n": len(rows),
                "meanFinalTrainAccuracy": _mean(rows, "final_train_accuracy"),
                "meanFinalValidationAccuracy": _mean(rows, "final_validation_accuracy"),
                "meanFinalTestAccuracy": _mean(rows, "final_test_accuracy"),
                "meanFinalParameters": _mean(rows, "final_parameters"),
            })
    candidate_groups = []
    for dataset in DATASETS:
        for kind in ("duplicate", "remove"):
            rows = [c for c in candidates if c["dataset"] == dataset and c["arm"] == "block_actions_on" and c["kind"] == kind]
            candidate_groups.append({
                "dataset": dataset,
                "kind": kind,
                "appearances": len(rows),
                "meanScore": _mean(rows, "score"),
                "selected": sum(bool(c["selected"]) for c in rows),
            })
    return {
        "experiment": "013_residual_block_multidataset",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generatedFrom": "saved history.pt and Experiment Board JSON artifacts",
        "status": {
            "completedRuns": sum(r["status"] == "completed" for r in records),
            "runningRuns": sum(r["status"] == "running" for r in records),
            "notStartedRuns": sum(r["status"] == "not_started" for r in records),
            "plannedRuns": len(records),
        },
        "runs": [{k: v for k, v in r.items() if k not in {"main", "history", "epochs", "actions"}} for r in records],
        "groupSummary": groups,
        "candidateSummary": candidate_groups,
        "actionCounts": [
            {
                "dataset": ds,
                "arm": arm,
                "completedRuns": sum(r["dataset"] == ds and r["arm"] == arm and r["status"] == "completed" for r in records),
                "runningRuns": sum(r["dataset"] == ds and r["arm"] == arm and r["status"] == "running" for r in records),
                "residualBlockExecutionsCompleted": sum(
                    1 for r in records if r["dataset"] == ds and r["arm"] == arm and r["status"] == "completed"
                    for action in r.get("actions", []) if "ResidualBlockAction" in action["name"] or "Residual Block Action" in action["name"]
                ),
                "residualBlockExecutionsRunning": sum(
                    1 for r in records if r["dataset"] == ds and r["arm"] == arm and r["status"] == "running"
                    for action in r.get("actions", []) if "ResidualBlockAction" in action["name"] or "Residual Block Action" in action["name"]
                ),
            }
            for ds in DATASETS for arm in ARMS
        ],
        "candidates": candidates,
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    records, candidates = load_records()
    _plot_training(records)
    _plot_parameters(records)
    _plot_test_accuracy(records)
    _plot_candidate_scores(candidates)
    SNAPSHOT.write_text(json.dumps(build_snapshot(records, candidates), indent=2), encoding="utf-8")
    print(f"Updated Experiment 013 charts and snapshot from {sum(r['status'] == 'completed' for r in records)} completed runs, "
          f"{sum(r['status'] == 'running' for r in records)} running, {sum(r['status'] == 'not_started' for r in records)} not started.")


if __name__ == "__main__":
    main()
