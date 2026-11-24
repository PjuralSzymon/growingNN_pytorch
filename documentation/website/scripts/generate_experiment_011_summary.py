"""Plot pooled final-accuracy means from the recorded Experiment 011 snapshot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SITE = Path(__file__).resolve().parents[1]
LABELS = {
    "stratified_random": "Stratified random",
    "el2n": "EL2N",
    "ccs": "CCS",
    "moderate": "Moderate",
    "moderate_coverage": "Moderate coverage",
}


def main():
    snapshot = SITE / "data/experiments/experiment-011-simulation-sets.json"
    runs = json.loads(snapshot.read_text(encoding="utf-8"))["runs"]
    groups = {name: [r for r in runs if r["algorithm"] == name] for name in LABELS}
    groups = {name: rows for name, rows in groups.items() if rows}
    mean = lambda rows, key: 100 * sum(r[key] for r in rows) / len(rows)
    names = sorted(groups, key=lambda name: mean(groups[name], "final_val"), reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    width = 0.36
    for offset, key, label, color in (
        (-width / 2, "final_val", "Validation", "#1d4ed8"),
        (width / 2, "final_test", "Test", "#d97706"),
    ):
        bars = ax.bar([i + offset for i in range(len(names))],
                      [mean(groups[name], key) for name in names],
                      width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xticks(range(len(names)), [
        f"{LABELS[name]}\n{sum(r['dataset'] == 'cifar10' for r in groups[name])} CIFAR-10 + "
        f"{sum(r['dataset'] == 'mnist' for r in groups[name])} MNIST"
        for name in names
    ], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean final accuracy (%)")
    ax.set_xlabel("Simulation-set generator · completed runs per dataset shown below each name")
    ax.set_title("Experiment 011: mean across all completed runs")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.text(0.5, 0.015, "Each run has equal weight. Dataset coverage differs between generators.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = SITE / "app/public/assets/experiments/011-summary-means.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
