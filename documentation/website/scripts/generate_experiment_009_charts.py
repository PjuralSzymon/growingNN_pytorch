"""Generate Experiment 009 charts from finished history.pt files, or from the JSON snapshot."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SITE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = SITE / "app" / "public" / "assets" / "experiments"
DEFAULT_SNAPSHOT = SITE / "data" / "experiments" / "experiment-009-best-reference.json"
RUN_DIR = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp009_best_reference"
)
SMALL_RUN_DIR = (
    SITE.parents[1]
    / "experiments"
    / "output"
    / "train_cifar10"
    / "runs"
    / "exp009_small_20"
)
SMALL_SNAPSHOT = SITE / "data" / "experiments" / "experiment-009-small-20.json"
DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
DATASET_LABEL = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist": "MNIST",
    "tiny_imagenet": "Tiny ImageNet",
}
SEED_COLORS = {1: "#1d4ed8", 2: "#d97706", 3: "#059669"}
SMALL_DATASETS = ("mnist", "cifar10", "cifar100")
REFERENCE_TEST = {
    "cifar10": 91.21,
    "cifar100": None,
    "mnist": None,
    "tiny_imagenet": None,
}


def _load_history(path: Path) -> dict[str, object]:
    """Read one history.pt into the lists the charts and the page use."""
    history = torch.load(path, map_location="cpu", weights_only=False)
    test = [float(value) for value in history["test_acc"]]
    params = [int(value) for value in history["param_count"]]
    actions = []
    for index, name in enumerate(history["action_name"]):
        actions.append(
            {
                "generation": int(history["action_generation"][index]),
                "name": str(name),
                "src": str(history["action_from"][index]),
                "dst": str(history["action_to"][index]),
                "val_before": float(history["action_val_acc_before"][index]),
                "val_after": float(history["action_val_acc_after"][index]),
            }
        )
    written = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "dataset": str(history["dataset"]),
        "seed": int(history["seed"]),
        "written": written,
        "train": [float(value) for value in history["train_acc"]],
        "val": [float(value) for value in history["val_acc"]],
        "test": test,
        "test_epoch": [10 * (index + 1) for index in range(len(test))],
        "params_epoch": params,
        "params_generation": [params[10 * index + 9] for index in range(len(test))],
        "final_train": float(history["train_acc"][-1]),
        "final_val": float(history["val_acc"][-1]),
        "final_test": test[-1],
        "params_start": params[0],
        "params_end": params[-1],
        "actions": actions,
    }


def load_measurement(run_dir: Path, datasets: tuple[str, ...] = DATASETS) -> dict[str, object]:
    """Load every finished dataset seed that has a history.pt file."""
    rows = []
    for dataset in datasets:
        for path in sorted((run_dir / dataset).glob("seed_*/history.pt")):
            rows.append(_load_history(path))
    return {"runs": rows}


def _rows(data: dict[str, object], dataset: str) -> list[dict[str, object]]:
    return [row for row in data["runs"] if row["dataset"] == dataset]


def _present(data: dict[str, object], datasets: tuple[str, ...] = DATASETS) -> list[str]:
    """Datasets that have at least one history, in the given order."""
    return [dataset for dataset in datasets if _rows(data, dataset)]


def _seeds(data: dict[str, object]) -> list[int]:
    """Seeds that appear in the loaded histories, in numeric order."""
    return [seed for seed in (1, 2, 3) if any(int(row["seed"]) == seed for row in data["runs"])]


def _panels(count: int, wide: tuple[float, float], grid: tuple[float, float]):
    """One row for up to three datasets, and a two-by-two grid for four."""
    if count >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(grid[0], grid[1]), dpi=140, sharex=True)
        return fig, list(axes.ravel())
    fig, axes = plt.subplots(1, count, figsize=(max(wide[0], 4.2 * count), wide[1]), dpi=140, sharex=True)
    return fig, [axes] if count == 1 else list(axes)


def plot_final_test(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-final-official-test.png",
    datasets: tuple[str, ...] = DATASETS,
) -> None:
    """Grouped bars of the last official test for each finished dataset."""
    present = _present(data, datasets)
    seeds = _seeds(data)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=140)
    width = 0.8 / max(len(seeds), 1)
    for offset, seed in enumerate(seeds):
        heights = []
        positions = []
        for index, dataset in enumerate(present):
            match = next((row for row in _rows(data, dataset) if int(row["seed"]) == seed), None)
            if match is None:
                continue
            heights.append(100.0 * float(match["final_test"]))
            positions.append(index + (offset - (len(seeds) - 1) / 2) * width)
        if heights:
            ax.bar(positions, heights, width=width, color=SEED_COLORS[seed], label=f"Seed {seed}")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([DATASET_LABEL[name] for name in present])
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color="#e2e8f0")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_train_histories(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-train-histories.png",
    mark_until: int = 120,
    datasets: tuple[str, ...] = DATASETS,
) -> None:
    """Per-epoch training accuracy for each dataset, with validation as a dashed line."""
    present = _present(data, datasets)
    fig, axes = _panels(len(present), (10.4, 3.8), (10.4, 6.8))
    for ax, dataset in zip(axes, present):
        for row in _rows(data, dataset):
            epochs = list(range(1, len(row["train"]) + 1))
            color = SEED_COLORS[int(row["seed"])]
            ax.plot(epochs, [100.0 * float(value) for value in row["train"]], color=color, linewidth=1.3, label=f"Seed {row['seed']} train")
            ax.plot(
                epochs,
                [100.0 * float(value) for value in row["val"]],
                color=color,
                linewidth=1.0,
                linestyle="--",
                label=f"Seed {row['seed']} validation",
            )
        for boundary in range(10, mark_until, 10):
            ax.axvline(boundary, color="#e2e8f0", linewidth=1, zorder=0)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_test_histories(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-test-by-generation.png",
    datasets: tuple[str, ...] = DATASETS,
) -> None:
    """Official test after each generation, one panel per dataset."""
    present = _present(data, datasets)
    fig, axes = _panels(len(present), (10.4, 3.8), (10.4, 6.6))
    for ax, dataset in zip(axes, present):
        for row in _rows(data, dataset):
            ax.plot(
                row["test_epoch"],
                [100.0 * float(value) for value in row["test"]],
                color=SEED_COLORS[int(row["seed"])],
                marker="o",
                linewidth=1.4,
                label=f"Seed {row['seed']}",
            )
        reference = REFERENCE_TEST[dataset]
        if reference is not None:
            ax.axhline(reference, color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(DATASET_LABEL[dataset])
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Official test accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_parameters(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-parameters.png",
    scale: int = 1_000_000,
    ylabel: str = "Parameters (millions)",
    datasets: tuple[str, ...] = DATASETS,
) -> None:
    """Parameter count at each generation end."""
    present = _present(data, datasets)
    fig, axes = _panels(len(present), (10.4, 3.8), (10.4, 6.6))
    for ax, dataset in zip(axes, present):
        for row in _rows(data, dataset):
            ax.plot(
                row["test_epoch"],
                [int(value) / scale for value in row["params_generation"]],
                color=SEED_COLORS[int(row["seed"])],
                marker="o",
                linewidth=1.4,
                label=f"Seed {row['seed']}",
            )
        ax.set_title(DATASET_LABEL[dataset])
        ax.grid(True, axis="y", color="#e2e8f0")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


SHORT_ACTION = {
    "AddSeqConvLayer": "Sequential conv",
    "AddResConvLayer": "Residual conv",
    "AddSeqLinearLayer": "Sequential linear",
    "AddResLinearLayer": "Residual linear",
    "AddSeqDropoutLayer": "Dropout",
    "AddNeurons": "Add neurons",
    "DelNeurons": "Delete neurons",
    "DelLayer": "Delete layer",
}


def _recovery_effects(data: dict[str, object], datasets: tuple[str, ...]) -> list[dict[str, object]]:
    """Accuracy change from the action's generation end to the next generation end."""
    effects = []
    for dataset in _present(data, datasets):
        for row in _rows(data, dataset):
            train = [float(value) for value in row["train"]]
            val = [float(value) for value in row["val"]]
            for action in row["actions"]:
                generation = int(action["generation"])
                before = 10 * (generation + 1) - 1
                after = before + 10
                if after >= len(val):
                    continue
                effects.append(
                    {
                        "dataset": dataset,
                        "seed": int(row["seed"]),
                        "generation": generation + 1,
                        "name": SHORT_ACTION.get(str(action["name"]), str(action["name"])),
                        "src": str(action.get("src", "")),
                        "dst": str(action.get("dst", "")),
                        "train": (train[after] - train[before]) * 100.0,
                        "val": (val[after] - val[before]) * 100.0,
                    }
                )
    return effects


def _scatter_strip(axis, values, center, color) -> None:
    count = len(values)
    if count == 0:
        return
    offsets = [0.0] if count == 1 else [-0.12 + 0.24 * index / (count - 1) for index in range(count)]
    axis.scatter(
        values,
        [center + offset for offset in offsets],
        color=color,
        edgecolor="#222222",
        linewidth=0.3,
        s=16,
        alpha=0.8,
        zorder=3,
    )


def _plot_effect_axis(axis, effects: list[dict[str, object]], names: list[str]) -> None:
    centers = list(range(len(names)))
    train = {name: [row["train"] for row in effects if row["name"] == name] for name in names}
    val = {name: [row["val"] for row in effects if row["name"] == name] for name in names}

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    axis.barh(
        [center - 0.18 for center in centers],
        [_mean(train[name]) for name in names],
        height=0.34,
        color="#3568a8",
        alpha=0.4,
        label="Training",
    )
    axis.barh(
        [center + 0.18 for center in centers],
        [_mean(val[name]) for name in names],
        height=0.34,
        color="#4f8a63",
        alpha=0.4,
        label="Validation",
    )
    for index, name in enumerate(names):
        _scatter_strip(axis, train[name], index - 0.18, "#3568a8")
        _scatter_strip(axis, val[name], index + 0.18, "#4f8a63")
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_yticks(centers, names)
    axis.grid(True, axis="x", color="#e2e8f0")


def plot_action_effects(
    data: dict[str, object],
    out_dir: Path,
    filename: str,
    datasets: tuple[str, ...] = SMALL_DATASETS,
) -> list[dict[str, object]]:
    """Mean and individual next-generation accuracy change, pooled and per dataset."""
    effects = _recovery_effects(data, datasets)
    names = sorted({str(row["name"]) for row in effects}, key=lambda name: sum(row["val"] for row in effects if row["name"] == name) / max(1, sum(row["name"] == name for row in effects)))
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    _plot_effect_axis(axis, effects, names)
    axis.set_xlabel("Accuracy change over the next generation (percentage points)")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / filename)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(datasets), figsize=(14.4, 4.6), sharex=True)
    for axis, dataset in zip(axes, datasets):
        subset = [row for row in effects if row["dataset"] == dataset]
        _plot_effect_axis(axis, subset, names)
        axis.set_title(DATASET_LABEL[dataset])
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("Accuracy change over the next generation (percentage points)")
    figure.tight_layout()
    figure.savefig(out_dir / filename.replace(".png", "-by-dataset.png"))
    plt.close(figure)
    return effects


def plot_action_counts(
    data: dict[str, object],
    out_dir: Path,
    filename: str = "009-small20-action-counts.png",
    datasets: tuple[str, ...] = SMALL_DATASETS,
) -> None:
    """How many times each action type was executed, summed across the three seeds."""
    effects = _recovery_effects(data, datasets)
    names = sorted({str(row["name"]) for row in effects})
    figure, axes = plt.subplots(1, len(datasets), figsize=(12.6, 4.2), sharey=True)
    for axis, dataset in zip(axes, datasets):
        counts = [sum(row["dataset"] == dataset and row["name"] == name for row in effects) for name in names]
        axis.barh(names, counts, color="#334155")
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_xlabel("Actions executed")
        axis.grid(True, axis="x", color="#e2e8f0")
    figure.tight_layout()
    figure.savefig(out_dir / filename)
    plt.close(figure)


def _matching_actions(actions, name: str, src: str, dst: str):
    found = []
    for action in actions:
        if type(action).__name__ != name or str(action.params[0]) != src:
            continue
        if name in ("AddNeurons", "DelNeurons"):
            if abs(float(action.params[1]) - float(dst)) < 1e-6:
                found.append(action)
        elif name == "DelLayer" or str(action.params[1]) == dst:
            found.append(action)
    return found


def _prefer_saved_detail(matches, name: str):
    """Pick one candidate when history stored only the endpoints."""
    if len(matches) == 1:
        return matches[0]
    if name == "AddSeqConvLayer":
        with_bn = [action for action in matches if len(action.params) > 5 and bool(action.params[5])]
        return with_bn[0] if with_bn else matches[0]
    if name == "AddSeqDropoutLayer":
        return min(matches, key=lambda action: abs(float(action.params[2].p) - 0.2))
    if name == "AddResLinearLayer":
        eye = [action for action in matches if "EYE" in str(action.params[3])]
        return eye[0] if eye else matches[0]
    return matches[0]


def write_architecture_images(
    data: dict[str, object],
    out_dir: Path,
    run_dir: Path,
    datasets: tuple[str, ...] = SMALL_DATASETS,
) -> None:
    """Replay saved actions and draw one reconstructed graph per finished seed."""
    import sys

    repo = SITE.parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from experiments.train_cifar10_exp009_best_reference import initial_graph
    from growingnn.actions.registry import generate_all_actions
    from growingnn.core.config import RunningConfig
    from growingnn.core.traced_model import TracedModel
    from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph

    shapes = {"mnist": (1, 1, 28, 28), "cifar10": (1, 3, 32, 32), "cifar100": (1, 3, 32, 32)}
    config = RunningConfig(generations=1, epochs=1, seq_conv_insert_before_global_pool=True)

    def replay(traced, steps: list[tuple[str, str, str]]) -> bool:
        if not steps:
            return True
        name, src, dst = steps[0]
        matches = _matching_actions(generate_all_actions(traced, config), name, src, dst)
        if name == "AddSeqDropoutLayer":
            matches = [_prefer_saved_detail(matches, name)] if matches else []
        future = " ".join(f"{step_src} {step_dst}" for _, step_src, step_dst in steps[1:])
        matches.sort(key=lambda action: 0 if len(action.params) > 3 and str(action.params[3]) in future else 1)
        for action in matches:
            backup = copy.deepcopy(traced.gm)
            try:
                action.execute(traced)
            except Exception:
                traced.gm = backup
                traced.invalidate()
                continue
            if replay(traced, steps[1:]):
                return True
            traced.gm = backup
            traced.invalidate()
        return False

    for dataset in datasets:
        images = []
        for row in _rows(data, dataset):
            seed = int(row["seed"])
            history_path = run_dir / dataset / f"seed_{seed}" / "history.pt"
            if not history_path.is_file():
                continue
            history = torch.load(history_path, map_location="cpu", weights_only=False)
            traced = TracedModel.create(initial_graph(dataset), shapes[dataset])
            steps = [
                (str(name), str(src), str(dst))
                for name, src, dst in zip(history["action_name"], history["action_from"], history["action_to"])
            ]
            if not replay(traced, steps):
                raise RuntimeError(f"Could not replay {dataset} seed {seed}")
            image_path = out_dir / f"009-small20-arch-{dataset}-seed{seed}"
            draw_filtered_fx_graph(traced.gm, str(image_path), fmt="png")
            images.append(image_path.with_suffix(".png"))
        if not images:
            continue
        figure, axes = plt.subplots(len(images), 1, figsize=(14, 4.2 * len(images)))
        if len(images) == 1:
            axes = [axes]
        for axis, image_path, row in zip(axes, images, _rows(data, dataset)):
            axis.imshow(plt.imread(image_path))
            axis.set_title(f"{DATASET_LABEL[dataset]} seed {row['seed']}")
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(out_dir / f"009-small20-arch-{dataset}.png")
        plt.close(figure)


def write_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write the Experiment 009 figures and refresh the snapshot from history.pt when those files exist."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    run_dir = run_dir or RUN_DIR
    finished = all((run_dir / dataset / "seed_1" / "history.pt").is_file() for dataset in DATASETS)
    if finished:
        data = load_measurement(run_dir)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_train_histories(data, out_dir)
    plot_final_test(data, out_dir)
    plot_test_histories(data, out_dir)
    plot_parameters(data, out_dir)


def write_small_starter_charts(
    out_dir: Path | None = None,
    snapshot_path: Path | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write the 20-generation one-convolution figures and refresh their snapshot."""
    out_dir = out_dir or DEFAULT_OUTPUT
    snapshot_path = snapshot_path or SMALL_SNAPSHOT
    run_dir = run_dir or SMALL_RUN_DIR
    finished = any((run_dir / dataset).glob("seed_*/history.pt") for dataset in SMALL_DATASETS)
    if finished:
        data = load_measurement(run_dir, SMALL_DATASETS)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_train_histories(
        data, out_dir, filename="009-small20-train-histories.png", mark_until=200, datasets=SMALL_DATASETS
    )
    plot_final_test(data, out_dir, filename="009-small20-final-official-test.png", datasets=SMALL_DATASETS)
    plot_test_histories(data, out_dir, filename="009-small20-test-by-generation.png", datasets=SMALL_DATASETS)
    plot_parameters(
        data,
        out_dir,
        filename="009-small20-parameters.png",
        scale=1_000,
        ylabel="Parameters (thousands)",
        datasets=SMALL_DATASETS,
    )
    effects = plot_action_effects(data, out_dir, "009-small20-action-types.png", SMALL_DATASETS)
    plot_action_counts(data, out_dir, datasets=SMALL_DATASETS)
    if finished:
        write_architecture_images(data, out_dir, run_dir, SMALL_DATASETS)
        for dataset in SMALL_DATASETS:
            subset = [row for row in effects if row["dataset"] == dataset]
            best = max(subset, key=lambda row: row["val"])
            worst = min(subset, key=lambda row: row["val"])
            print(
                dataset,
                "best",
                f"{best['val']:+.2f}",
                best["name"],
                best["seed"],
                best["generation"],
                "worst",
                f"{worst['val']:+.2f}",
                worst["name"],
                worst["seed"],
                worst["generation"],
            )


if __name__ == "__main__":
    write_charts()
    write_small_starter_charts()
