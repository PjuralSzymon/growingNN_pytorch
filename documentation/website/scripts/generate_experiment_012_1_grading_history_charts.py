"""Generate Experiment 013 accuracy and parameter history figures from board metrics."""

from __future__ import annotations

import json
import pickle
import subprocess
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[3]
# Predecessor runs stay in the exp011 folder. The current runner writes a new directory.
RUNS = ROOT / "experiments/output/train_cifar10/runs/exp011_residual_block_action_grading"
ASSETS = ROOT / "documentation/website/app/public/assets/experiments"
DATASETS = ("cifar10", "cifar100")
ARMS = ("block_actions_on", "block_actions_off")
INITIALIZATIONS = ("copy", "quasi_identity", "identity_residual_zero")
INIT_TITLES = {"copy": "Copied weights", "quasi_identity": "Quasi-identity", "identity_residual_zero": "Residual zero"}
COLORS = {1: "#3568a8", 2: "#e4572e", 3: "#2a9d68"}
ARM_TITLES = {"block_actions_on": "Block actions enabled", "block_actions_off": "Block actions disabled"}


def _load_history(path: Path) -> dict:
    """Load the plain metric lists stored in this experiment's torch.save archive."""
    with zipfile.ZipFile(path) as archive:
        return pickle.loads(archive.read("history/data.pkl"))


def _initialization(parts: tuple[str, ...]) -> str:
    return next((part.removeprefix("weight_init_") for part in parts if part.startswith("weight_init_")), "copy")


def load_runs() -> list[dict]:
    runs = []
    for history_path in sorted(RUNS.rglob("history.pt")):
        if "_failed_" in str(history_path):
            continue
        relative = history_path.relative_to(RUNS)
        dataset = "cifar100" if "cifar100" in relative.parts else "cifar10"
        arm = next((part for part in relative.parts if part in ARMS), None)
        if arm is None:
            continue
        seed = int(next(part.removeprefix("seed_") for part in relative.parts if part.startswith("seed_")))
        main_path = history_path.parent / "board" / "main.json"
        main = json.loads(main_path.read_text(encoding="utf-8")) if main_path.exists() else {}
        if main and main.get("status") != "completed":
            continue
        history = _load_history(history_path)
        epochs_per_generation = int(history["epochs_per_generation"])
        epoch_generations = [
            generation
            for generation in history["generation"]
            for _ in range(epochs_per_generation)
        ]
        epochs = [
            {"globalEpoch": index, "generation": generation,
             "trainAcc": history["train_acc"][index], "valAcc": history["val_acc"][index],
             "paramCount": history["param_count"][index]}
            for index, generation in enumerate(epoch_generations)
        ]
        runs.append({"dataset": dataset, "arm": arm, "init": _initialization(relative.parts),
                     "seed": seed, "path": history_path, "epochs": epochs, "history": history,
                     "generation_timeline": main.get("generationTimeline", [])})
    return runs


def plot_accuracy(runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
    for row, dataset in enumerate(DATASETS):
        for col, initialization in enumerate(INITIALIZATIONS):
            ax = axes[row, col]
            selected = [run for run in runs if run["dataset"] == dataset and run["arm"] == "block_actions_on" and run["init"] == initialization]
            for run in sorted(selected, key=lambda item: item["seed"]):
                epochs = run["epochs"]
                x = [int(epoch["globalEpoch"]) + 1 for epoch in epochs]
                color = COLORS[run["seed"]]
                ax.plot(x, [100 * epoch["trainAcc"] for epoch in epochs], color=color,
                        linewidth=1.7, label=f"Seed {run['seed']} train")
                ax.plot(x, [100 * epoch["valAcc"] for epoch in epochs], color=color,
                        linewidth=1.5, linestyle="--", label=f"Seed {run['seed']} validation")
            # Disabled controls are shared across initialization modes and plotted in gray.
            controls = [run for run in runs if run["dataset"] == dataset and run["arm"] == "block_actions_off" and run["init"] == "copy"]
            for run in sorted(controls, key=lambda item: item["seed"]):
                epochs = run["epochs"]
                x = [int(epoch["globalEpoch"]) + 1 for epoch in epochs]
                shade = {1: "#777777", 2: "#999999", 3: "#bbbbbb"}[run["seed"]]
                ax.plot(x, [100 * epoch["trainAcc"] for epoch in epochs], color=shade,
                        linewidth=0.9, alpha=0.55, label="Disabled control" if run["seed"] == 1 else None)
                ax.plot(x, [100 * epoch["valAcc"] for epoch in epochs], color=shade,
                        linewidth=0.9, linestyle="--", alpha=0.55)
            for boundary in range(5, 50, 5):
                ax.axvline(boundary, color="#aab2bd", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_title(f"{dataset.upper()} — {INIT_TITLES[initialization]}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xlim(1, 50)
            ax.set_ylim(0, 80)
            ax.grid(alpha=0.22)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Experiment 013: per-epoch histories by duplicate initialization", fontsize=15)
    fig.text(0.5, 0.015, "Seed colors show enabled runs; gray traces are the same completed actions-disabled controls in every panel. Solid=train, dashed=validation. Dotted lines mark generation boundaries.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(ASSETS / "013-training-histories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_parameters(runs: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for row, dataset in enumerate(DATASETS):
        for col, initialization in enumerate(INITIALIZATIONS):
            ax = axes[row, col]
            selected = [run for run in runs if run["dataset"] == dataset and run["arm"] == "block_actions_on" and run["init"] == initialization]
            for run in sorted(selected, key=lambda item: item["seed"]):
                final_epoch_by_gen: dict[int, dict] = {}
                for epoch in run["epochs"]:
                    final_epoch_by_gen[int(epoch["generation"])] = epoch
                generations = sorted(final_epoch_by_gen)
                counts = [int(final_epoch_by_gen[generation]["paramCount"]) for generation in generations]
                ax.plot(generations, counts, color=COLORS[run["seed"]], marker="o",
                        linewidth=1.8, markersize=4, label=f"Seed {run['seed']}")
            controls = [run for run in runs if run["dataset"] == dataset and run["arm"] == "block_actions_off" and run["init"] == "copy"]
            for run in sorted(controls, key=lambda item: item["seed"]):
                final_epoch_by_gen: dict[int, dict] = {}
                for epoch in run["epochs"]:
                    final_epoch_by_gen[int(epoch["generation"])] = epoch
                generations = sorted(final_epoch_by_gen)
                counts = [int(final_epoch_by_gen[generation]["paramCount"]) for generation in generations]
                shade = {1: "#777777", 2: "#999999", 3: "#bbbbbb"}[run["seed"]]
                ax.plot(generations, counts, color=shade, marker=".", linewidth=0.9, alpha=0.55,
                        label="Disabled control" if run["seed"] == 1 else None)
            ax.set_title(f"{dataset.upper()} — {INIT_TITLES[initialization]}")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Parameter count")
            ax.set_xticks(range(10))
            ax.grid(alpha=0.22)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Experiment 013: parameter count by initialization and generation", fontsize=15)
    fig.text(0.5, 0.015, "Seed colors show enabled runs; gray traces are the same completed actions-disabled controls in every panel.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(ASSETS / "013-parameter-histories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _action_label(name: str) -> str:
    labels = {
        "AddSeqLinearLayer": "Add sequential linear",
        "AddSeqConvLayer": "Add sequential convolution",
        "AddSeqDropoutLayer": "Add dropout",
        "AddResConvLayer": "Add residual convolution",
        "AddResLinearLayer": "Add residual linear",
        "DeleteLayer": "Delete layer",
        "AddNeurons": "Add neurons",
        "DeleteNeurons": "Delete neurons",
        "DelLayer": "Delete layer",
        "DelNeurons": "Delete neurons",
        "ResidualBlockAction": "Residual block action",
    }
    return labels.get(name, name)


def load_action_effects() -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Return post-action next-generation accuracy changes in percentage points."""
    effects: dict[str, dict[str, list[tuple[float, float]]]] = {
        dataset: {} for dataset in DATASETS
    }
    for path in sorted(RUNS.rglob("history.pt")):
        if "_failed_" in str(path):
            continue
        status_path = path.parent / "board" / "main.json"
        if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") != "completed":
            continue
        history = _load_history(path)
        relative = path.relative_to(RUNS)
        dataset = "cifar100" if "cifar100" in relative.parts else "cifar10"
        generations: dict[int, dict[str, float]] = {}
        for index, generation in enumerate(history["generation"]):
            generations[int(generation)] = {
                "train": float(history["train_acc"][index]),
                "val": float(history["val_acc"][index]),
            }
        for generation, action_name in zip(
            history["action_generation"], history["action_name"], strict=True
        ):
            generation = int(generation)
            if generation not in generations or generation + 1 not in generations:
                continue
            label = _action_label(str(action_name))
            pair = effects[dataset].setdefault(label, [])
            pair.append((
                100 * (generations[generation + 1]["train"] - generations[generation]["train"]),
                100 * (generations[generation + 1]["val"] - generations[generation]["val"]),
            ))
    return effects


def plot_action_effects() -> None:
    effects = load_action_effects()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        grouped = effects[dataset]
        names = sorted(grouped)
        y = list(range(len(names)))
        ax.barh(
            [position - 0.18 for position in y],
            [sum(row[0] for row in grouped[name]) / len(grouped[name]) for name in names],
            height=0.34, color="#3568a8", alpha=0.4, label="Training accuracy",
        )
        ax.barh(
            [position + 0.18 for position in y],
            [sum(row[1] for row in grouped[name]) / len(grouped[name]) for name in names],
            height=0.34, color="#4f8a63", alpha=0.4, label="Validation accuracy",
        )
        for position, name in enumerate(names):
            rows = grouped[name]
            count = len(rows)
            offsets = [0.0] if count == 1 else [-0.1 + 0.2 * i / (count - 1) for i in range(count)]
            ax.scatter([row[0] for row in rows], [position - 0.18 + offset for offset in offsets],
                       color="#3568a8", edgecolor="#222", linewidth=0.3, s=22, zorder=3)
            ax.scatter([row[1] for row in rows], [position + 0.18 + offset for offset in offsets],
                       color="#4f8a63", edgecolor="#222", linewidth=0.3, s=22, zorder=3)
        ax.axvline(0, color="#333", linewidth=0.9)
        ax.set_yticks(y, [f"{name} (n={len(grouped[name])})" for name in names])
        ax.invert_yaxis()
        ax.set_title(dataset.upper())
        ax.set_xlabel("Accuracy change over next generation (percentage points)")
        ax.grid(axis="x", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Experiment 013: training and validation change after selected actions")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(ASSETS / "013-action-effects.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_duplicate_impacts(runs: list[dict]) -> None:
    """Show the three selected block duplicates and the next-generation changes."""
    events = []
    for run in runs:
        history = run["history"]
        for timeline_row in run["generation_timeline"]:
            action = timeline_row.get("actionExecuted") or {}
            action_name = str(action.get("action", ""))
            if "Residual Block Action" not in action_name or "duplicate" not in action_name.lower():
                continue
            generation = int(timeline_row["generation"])
            epochs_per_generation = int(history["epochs_per_generation"])
            before = (generation + 1) * epochs_per_generation - 1
            after = (generation + 2) * epochs_per_generation - 1
            if after >= len(history["train_acc"]):
                continue
            events.append({
                "run": run, "generation": generation,
                "before": {
                    "train": 100 * float(history["train_acc"][before]),
                    "val": 100 * float(history["val_acc"][before]),
                    "params": int(history["param_count"][before]),
                },
                "after": {
                    "train": 100 * float(history["train_acc"][after]),
                    "val": 100 * float(history["val_acc"][after]),
                    "params": int(history["param_count"][after]),
                },
            })
    events.sort(key=lambda e: (e["run"]["dataset"], e["run"]["init"], e["run"]["seed"]))
    if not events:
        return

    fig, axes = plt.subplots(2, len(events), figsize=(6.1 * len(events), 9), squeeze=False)
    for col, event in enumerate(events):
        run = event["run"]
        history = run["history"]
        ax = axes[0, col]
        x = list(range(1, len(history["train_acc"]) + 1))
        ax.plot(x, [100 * float(v) for v in history["train_acc"]],
                color="#3568a8", linewidth=1.6, label="Train")
        ax.plot(x, [100 * float(v) for v in history["val_acc"]],
                color="#4f8a63", linewidth=1.6, linestyle="--", label="Validation")
        epochs_per_generation = int(history["epochs_per_generation"])
        boundary = (event["generation"] + 1) * epochs_per_generation + 0.5
        ax.axvline(boundary, color="#7c3aed", linewidth=2, linestyle=":")
        ax.axvspan(boundary, boundary + epochs_per_generation, color="#7c3aed", alpha=0.10, label="First generation after duplicate")
        ax.annotate("duplicate selected", xy=(boundary, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(5, -5), textcoords="offset points", color="#6d28d9", fontsize=8,
                    rotation=90, va="top")
        ax.set_title(f"{run['dataset'].upper()} · {INIT_TITLES[run['init']]} · seed {run['seed']}\nAfter generation {event['generation']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(1, len(history["train_acc"]))
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[1, col]
        categories = ["Train", "Validation"]
        before_values = [event["before"]["train"], event["before"]["val"]]
        after_values = [event["after"]["train"], event["after"]["val"]]
        positions = list(range(len(categories)))
        width = 0.34
        ax.bar([p - width / 2 for p in positions], before_values, width, color="#aab2bd", label=f"Before (gen {event['generation']})")
        ax.bar([p + width / 2 for p in positions], after_values, width, color="#7c3aed", label=f"After (gen {event['generation'] + 1})")
        for i, (old, new) in enumerate(zip(before_values, after_values, strict=True)):
            ax.text(i, max(old, new) + 1.0, f"{new-old:+.2f} pp", ha="center", fontsize=9)
        ax.set_xticks(positions, categories)
        ax.set_ylabel("Final epoch accuracy (%)")
        ax.set_ylim(0, 85)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8, frameon=False)
        params_delta = event["after"]["params"] - event["before"]["params"]
        ax.text(0.5, -0.24,
                f"Parameters: {event['before']['params']:,} → {event['after']['params']:,} ({params_delta:+,})",
                transform=ax.transAxes, ha="center", fontsize=9)

    fig.suptitle("Experiment 013: observed changes after selected residual-block duplicates", fontsize=15)
    fig.text(0.5, 0.015,
             "Before/after are final-epoch metrics for adjacent generations. Accuracy changes include continued training and other effects; they are not causal estimates.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(ASSETS / "013-block-duplicate-impact.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_graph(pdf_path: Path, prefix: Path) -> Path:
    subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "125", str(pdf_path), str(prefix)],
        check=True, capture_output=True,
    )
    rendered = prefix.with_suffix(".png")
    image = Image.open(rendered).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    diff = ImageChops.difference(image, background).convert("L").point(lambda value: 255 if value > 28 else 0)
    bounds = diff.getbbox()
    if bounds:
        padding = 28
        left, top, right, bottom = bounds
        bounds = (max(0, left - padding), max(0, top - padding),
                  min(image.width, right + padding), min(image.height, bottom + padding))
        image.crop(bounds).save(rendered)
    return rendered


def plot_duplicate_structure_changes(runs: list[dict]) -> None:
    """Write a readable before/after graph pair for every selected duplicate."""
    events = []
    for run in runs:
        for row in run["generation_timeline"]:
            selected = row.get("actionExecuted") or {}
            action = selected.get("action", "")
            if "Residual Block Action" in action and "duplicate" in action.lower():
                events.append((run, int(row["generation"])))
    events.sort(key=lambda item: (item[0]["dataset"], item[0]["init"], item[0]["seed"]))

    with tempfile.TemporaryDirectory(prefix="exp011_block_graphs_") as temp_dir:
        temp = Path(temp_dir)
        for index, (run, generation) in enumerate(events):
            board = run["path"].parent / "board"
            before_pdf = board / "graphs" / f"gen_{generation}_simplified.pdf"
            after_pdf = board / "graphs" / f"gen_{generation + 1}_simplified.pdf"
            fig, axes = plt.subplots(1, 2, figsize=(15, 3.6))
            for ax, graph_path, label, graph_generation in (
                (axes[0], before_pdf, "Before duplicate", generation),
                (axes[1], after_pdf, "After duplicate", generation + 1),
            ):
                if not graph_path.exists():
                    ax.text(0.5, 0.5, f"Missing graph: {graph_path.name}", ha="center", va="center")
                else:
                    rendered = _render_graph(graph_path, temp / f"event{index}_{label.split()[0].lower()}")
                    ax.imshow(plt.imread(rendered))
                ax.set_title(f"{label} · generation {graph_generation}")
                ax.axis("off")
            fig.suptitle(
                f"{run['dataset'].upper()} · {INIT_TITLES[run['init']]} · seed {run['seed']} · block duplicate after generation {generation}",
                fontsize=13,
            )
            fig.text(0.5, 0.015,
                     "The copied residual branch appears as block_conv_block_0 and block_bn_block_0 in the after graph.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 0.90))
            dataset = run["dataset"]
            init = run["init"]
            seed = run["seed"]
            fig.savefig(ASSETS / f"013-block-duplicate-structure-{dataset}-{init}-seed{seed}.png",
                        dpi=170, bbox_inches="tight")
            plt.close(fig)


def plot_initialization_outcomes(runs: list[dict]) -> None:
    """Plot final test outcomes by initialization, retaining the shared controls."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    init_colors = {"copy": "#3568a8", "quasi_identity": "#e9a23b", "identity_residual_zero": "#2a9d68"}
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for x, initialization in enumerate(INITIALIZATIONS):
            selected = [r for r in runs if r["dataset"] == dataset and r["arm"] == "block_actions_on" and r["init"] == initialization]
            values = [100 * float(r["history"]["test_acc"][-1]) for r in selected]
            if values:
                jitter = [x + (i - (len(values) - 1) / 2) * 0.08 for i in range(len(values))]
                ax.scatter(jitter, values, color=init_colors[initialization], label=INIT_TITLES[initialization], zorder=3)
                ax.plot([x - 0.18, x + 0.18], [sum(values) / len(values)] * 2, color="#222", linewidth=2)
        controls = [r for r in runs if r["dataset"] == dataset and r["arm"] == "block_actions_off" and r["init"] == "copy"]
        values = [100 * float(r["history"]["test_acc"][-1]) for r in controls]
        if values:
            x = len(INITIALIZATIONS)
            jitter = [x + (i - (len(values) - 1) / 2) * 0.08 for i in range(len(values))]
            ax.scatter(jitter, values, color="#888", marker="s", label="Actions disabled", zorder=3)
            ax.plot([x - 0.18, x + 0.18], [sum(values) / len(values)] * 2, color="#222", linewidth=2)
        ax.set_xticks(range(4), ["Copy", "Quasi-identity", "Residual-zero", "Disabled"])
        ax.set_ylabel("Final test accuracy (%)")
        ax.set_title(dataset.upper())
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Experiment 013: final test accuracy by block initialization")
    fig.text(0.5, 0.015, "Dots are individual seeds; black bars mark means. Disabled controls are shared across initializer comparisons.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(ASSETS / "013-initialization-test-comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    plot_accuracy(runs)
    plot_parameters(runs)
    plot_action_effects()
    plot_duplicate_impacts(runs)
    plot_duplicate_structure_changes(runs)
    plot_initialization_outcomes(runs)
    print(f"Wrote Experiment 013 history figures to {ASSETS}")


if __name__ == "__main__":
    main()
