"""Generate Experiment 013 score and final-test plots from its run artifacts."""

from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
# Predecessor runs stay in the exp011 folder. The current runner writes a new directory.
RUNS = ROOT / "experiments/output/train_cifar10/runs/exp011_residual_block_action_grading"
ASSETS = ROOT / "documentation/website/app/public/assets/experiments"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[float]] = {}
    block_groups: dict[tuple[str, str, str], list[float]] = {}
    for path in sorted(RUNS.rglob("board/simulations/simulation_gen_*.json")):
        if "_failed_" in str(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "block_actions_on" not in path.parts:
            continue
        relative = path.relative_to(RUNS)
        dataset = "cifar100" if "cifar100" in relative.parts else "cifar10"
        init = next((part.removeprefix("weight_init_") for part in relative.parts if part.startswith("weight_init_")), "copy")
        for row in payload.get("candidates", []):
            action = row.get("action", "")
            name = row.get("name", "")
            if "Residual Block Action" in action:
                block_kind = "duplicate" if "duplicate" in action.lower() else "remove"
                name = "Block duplicate" if block_kind == "duplicate" else "Block remove"
                if row.get("score") is not None:
                    block_groups.setdefault((dataset, init, block_kind), []).append(float(row["score"]))
            elif name.endswith(" Action"):
                name = name[:-7]
            if row.get("score") is not None:
                groups.setdefault((dataset, name), []).append(float(row["score"]))

    preferred = ["Block duplicate", "Block remove", "Add Res Conv Layer", "Add Res Linear Layer",
                 "Add Seq Conv Layer", "Add Seq Linear Layer", "Add Seq Dropout Layer", "Delete Layer"]
    available = {name for _, name in groups}
    labels = [name for name in preferred if name in available]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, dataset in zip(axes, ("cifar10", "cifar100")):
        present = [name for name in labels if (dataset, name) in groups]
        box = ax.boxplot([groups[(dataset, name)] for name in present], tick_labels=present,
                         patch_artist=True, showmeans=True)
        for patch, name in zip(box["boxes"], present):
            patch.set_facecolor("#7c3aed" if "Block" in name else "#60a5fa")
            patch.set_alpha(0.65)
        ax.set_title(dataset.upper())
        ax.set_ylabel("MCTS mean rollout score (validation accuracy)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Experiment 013: enabled-arm candidate grades")
    fig.tight_layout()
    fig.savefig(ASSETS / "013-candidate-grades.png", dpi=170)
    plt.close(fig)

    initializations = ["copy", "quasi_identity", "identity_residual_zero"]
    init_labels = ["Copy", "Quasi-identity", "Residual-zero"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, dataset in zip(axes, ("cifar10", "cifar100")):
        positions, values, colors = [], [], []
        for index, initialization in enumerate(initializations):
            for offset, kind, color in ((-0.16, "duplicate", "#7c3aed"), (0.16, "remove", "#94a3b8")):
                rows = block_groups.get((dataset, initialization, kind), [])
                if rows:
                    positions.append(index + offset)
                    values.append(rows)
                    colors.append(color)
        if values:
            box = ax.boxplot(values, positions=positions, widths=0.24, patch_artist=True,
                             showmeans=True, manage_ticks=False)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.68)
        ax.set_xticks(range(3), init_labels)
        ax.set_title(dataset.upper())
        ax.set_ylabel("Candidate rollout score (validation accuracy)")
        ax.grid(axis="y", alpha=0.24)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#7c3aed", label="Duplicate"),
                        Patch(facecolor="#94a3b8", label="Remove")],
               loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=2, frameon=False)
    fig.suptitle("Experiment 013: residual-block candidate grades by duplicate initializer")
    fig.text(0.5, 0.02, "Boxes pool candidate appearances across seeds and generations; each candidate received one or two visits.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(ASSETS / "013-block-grades-by-initialization.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    tests: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for path in sorted(RUNS.rglob("history.pt")):
        if "_failed_" in str(path):
            continue
        with zipfile.ZipFile(path) as archive:
            history = pickle.loads(archive.read("history/data.pkl"))
        relative = path.relative_to(RUNS)
        dataset = "cifar100" if "cifar100" in relative.parts else "cifar10"
        arm = "Enabled" if "block_actions_on" in relative.parts else "Disabled"
        init = next((part.removeprefix("weight_init_") for part in relative.parts if part.startswith("weight_init_")), "copy")
        arm = f"{arm} · {init}" if arm == "Enabled" else arm
        tests.setdefault((dataset, arm), []).append((int(history["seed"]), float(history["test_acc"][-1])))
    arms = ["Enabled · copy", "Enabled · quasi_identity", "Enabled · identity_residual_zero", "Disabled"]
    colors = {"Enabled · copy": "#3568a8", "Enabled · quasi_identity": "#e9a23b",
              "Enabled · identity_residual_zero": "#2a9d68", "Disabled": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dataset in zip(axes, ("cifar10", "cifar100")):
        for x, arm in enumerate(arms):
            vals = tests.get((dataset, arm), [])
            ys = [value for _, value in vals]
            if ys:
                ax.scatter([x + (i - (len(ys)-1)/2)*0.06 for i in range(len(ys))], [100*v for v in ys],
                           color=colors[arm], label=arm, zorder=3)
                ax.plot([x-0.16, x+0.16], [100*sum(ys)/len(ys)]*2, color="#333333", linewidth=2)
        ax.set_xticks(range(len(arms)), ["Copy", "Quasi-id", "Residual-zero", "Disabled"], rotation=15)
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_title(dataset.upper())
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Final test accuracy by duplicate initialization; bars mark means")
    fig.tight_layout()
    fig.savefig(ASSETS / "013-final-test.png", dpi=170)
    plt.close(fig)
    print(f"Wrote Experiment 013 charts to {ASSETS}")


if __name__ == "__main__":
    main()
