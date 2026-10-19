"""
Experiment 008 driver — CIFAR-10 generation loop only.

Experiment 007 is unchanged. It still compares the original recipe with the
added package and its finish-chunk stage.

This experiment does not run that finish stage. Training is train_generations:
twelve generations of ten epochs, and one sequential convolution after every
generation except the last. The learning rate is a cosine from 0.1 to 0.001
across those epochs. The starter has 128 channels. Cutout is on the training
loader only. Seeds are 110, 111, and 112.

The official 10,000-image test set is logged. Search uses the 5,000-image
validation slice of the training set.

Phase 1 allows only AddSeqConvLayer:

    pipenv run python -u experiments/train_cifar10_exp008_generation_only.py

Phase 2 keeps that recipe and changes only the allowed action set. The groups
are all, grow, and seq_res. Each group runs seeds 110, 111, and 112.

    pipenv run python -u experiments/train_cifar10_exp008_generation_only.py --phase 2

Phase 3 keeps the all-action list and lengthens the search budget to 100
seconds, 8 epochs, and 2,000 images. Seeds are 110, 111, and 112.

    pipenv run python -u experiments/train_cifar10_exp008_generation_only.py --phase 3

Phase 4 keeps that same all-action list and Monte Carlo. It compares simulation
epoch counts 1, 2, and 4. Each grade uses a 300 second budget, all 45,000
augmented training images, the full validation slice, and no extra rollout
actions. Seeds are 110, 111, and 112.

    pipenv run python -u experiments/train_cifar10_exp008_generation_only.py --phase 4

Phase 5 keeps the phase 4 epoch counts, budget, validation slice, and rollout
depth. The simulation training set is the first 5,000 clean training images.
Those images have no crop, flip, or cutout.

    pipenv run python -u experiments/train_cifar10_exp008_generation_only.py --phase 5
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config as project_config
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10 import DATA_DIR, Cifar10Data, _BatchLoader, _CudaCIFAR
from experiments.train_cifar10_exp007_test90_package import (
    added_hyperparameters,
    build_model,
    running_config_for_group,
)
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

RUNS_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp008_generation_only"
PHASE2_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp008_generation_only_phase2"
PHASE3_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp008_generation_only_phase3"
PHASE4_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp008_generation_only_phase4"
PHASE5_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp008_generation_only_phase5"
SEEDS = (110, 111, 112)
PHASE2_GROUPS = ("all", "grow", "seq_res")
PHASE3_SIMULATION = (100.0, 8, 2000)
PHASE4_SIMULATIONS = (
    (300.0, 1, 45000),
    (300.0, 2, 45000),
    (300.0, 4, 45000),
)
PHASE4_ROLLOUT_DEPTH = 0
PHASE5_CLEAN_PREFIX = 5000
PHASE5_SIMULATIONS = (
    (300.0, 1, PHASE5_CLEAN_PREFIX),
    (300.0, 2, PHASE5_CLEAN_PREFIX),
    (300.0, 4, PHASE5_CLEAN_PREFIX),
)
PHASE5_ROLLOUT_DEPTH = 0
_ACTION_FLAGS = (
    "ACTIONS_ENABLE_ADD_SEQ_LAYER",
    "ACTIONS_ENABLE_ADD_RES_LAYER",
    "ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER",
    "ACTIONS_ENABLE_ADD_RES_CONV_LAYER",
    "ACTIONS_ENABLE_ADD_NEURONS_11",
    "ACTIONS_ENABLE_ADD_NEURONS_15",
    "ACTIONS_ENABLE_ADD_NEURONS_20",
    "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01",
    "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_02",
    "ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05",
    "ACTIONS_ENABLE_DEL_LAYER",
    "ACTIONS_ENABLE_DEL_NEURONS_01",
    "ACTIONS_ENABLE_DEL_NEURONS_05",
    "ACTIONS_ENABLE_DEL_NEURONS_09",
)
_GROW_FLAGS = (
    "ACTIONS_ENABLE_ADD_SEQ_LAYER",
    "ACTIONS_ENABLE_ADD_RES_LAYER",
    "ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER",
    "ACTIONS_ENABLE_ADD_RES_CONV_LAYER",
    "ACTIONS_ENABLE_ADD_NEURONS_11",
    "ACTIONS_ENABLE_ADD_NEURONS_15",
    "ACTIONS_ENABLE_ADD_NEURONS_20",
)
_SEQ_RES_FLAGS = (
    "ACTIONS_ENABLE_ADD_SEQ_LAYER",
    "ACTIONS_ENABLE_ADD_RES_LAYER",
    "ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER",
    "ACTIONS_ENABLE_ADD_RES_CONV_LAYER",
)
GENERATIONS = 12
EPOCHS = 10
CHANNELS = 128
HIDDEN = 256
LABEL_SMOOTHING = 0.1
_EPOCH_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "param_count")
_ACTION_KEYS = (
    "action_generation",
    "action_name",
    "action_from",
    "action_to",
    "action_val_acc_before",
    "action_val_acc_after",
)


class CosineDecayScheduler:
    """Cosine from 0.1 down to a floor. A structure change does not reset the epoch counter."""

    def __init__(self, total_epochs: int, floor: float = 0.001) -> None:
        self.global_epoch = 0
        self.total_epochs = max(int(total_epochs), 1)
        self.floor = float(floor)
        self._global_schedule_progress_frozen = False

    def alpha_scheduler(self, _i: int, _iterations: int) -> float:
        epoch = min(self.global_epoch, self.total_epochs - 1)
        if not self._global_schedule_progress_frozen:
            self.global_epoch += 1
        progress = epoch / max(self.total_epochs - 1, 1)
        return self.floor + 0.5 * (0.1 - self.floor) * (1.0 + math.cos(math.pi * progress))

    def freeze_global_schedule_progress(self):
        """Keep simulation scoring from advancing the training epoch counter."""
        from contextlib import contextmanager

        scheduler = self

        @contextmanager
        def _freeze():
            previous = scheduler._global_schedule_progress_frozen
            scheduler._global_schedule_progress_frozen = True
            try:
                yield scheduler
            finally:
                scheduler._global_schedule_progress_frozen = previous

        return _freeze()

    def structure_changed(self) -> None:
        return None

    def learning_rate_config_board_labels(self) -> tuple[str, float]:
        return "CosineDecay", 0.1


class _BatchCutout:
    """Erase rectangles on batches that are already on the CUDA loader."""

    def __init__(self, loader: DataLoader, erasing: transforms.RandomErasing) -> None:
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self._loader = loader
        self._erasing = erasing

    def __iter__(self):
        for images, targets in self._loader:
            erased = images.clone()
            for index in range(erased.shape[0]):
                erased[index] = self._erasing(erased[index])
            yield erased, targets

    def __len__(self) -> int:
        return len(self._loader)


def with_cutout(loader: DataLoader) -> DataLoader | _BatchCutout:
    """Apply RandomErasing on the augmented training batches only."""
    erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    if getattr(loader, "_shuffle", None) is not None and hasattr(loader.dataset, "images"):
        return _BatchCutout(loader, erasing)
    base = loader.dataset

    class _Cutout(Dataset):
        def __len__(self) -> int:
            return len(base)

        def __getitem__(self, index: int):
            image, target = base[index]
            return erasing(image), target

    return DataLoader(_Cutout(), batch_size=loader.batch_size, shuffle=True)


def enabled_action_flags(group: str) -> tuple[str, ...]:
    """Return the action flags that stay on for one Experiment 008 group."""
    if group == "seq_conv":
        return ("ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER",)
    if group == "all":
        return _ACTION_FLAGS
    if group == "grow":
        return _GROW_FLAGS
    if group == "seq_res":
        return _SEQ_RES_FLAGS
    raise ValueError(f"unknown action group {group}")


def apply_action_group(cfg, group: str) -> None:
    """Turn action flags on or off. Every other training setting stays as passed in."""
    enabled = set(enabled_action_flags(group))
    for flag in _ACTION_FLAGS:
        setattr(cfg, flag, flag in enabled)


def apply_simulation_budget(cfg, simulation: tuple[float, int, int]) -> None:
    """Replace how long Monte Carlo looks. Action flags and the training recipe stay as they are."""
    time_s, epochs, set_size = simulation
    cfg.simulation_set_size = int(set_size)
    cfg.simulation_scheduler = AlwaysSimulationScheduler(float(time_s), int(epochs))


@contextmanager
def rollout_depth(depth: int | None):
    """Use one Monte Carlo rollout depth for the block, then restore the previous value."""
    if depth is None:
        yield
        return
    previous = project_config.MCTS_ROLLOUT_DEPTH
    project_config.MCTS_ROLLOUT_DEPTH = int(depth)
    try:
        yield
    finally:
        project_config.MCTS_ROLLOUT_DEPTH = previous


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select a phase and an optional seed subset. Phase 2 also accepts action groups."""
    parser = argparse.ArgumentParser(description="Experiment 008: CIFAR-10 generation loop only.")
    parser.add_argument("--phase", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    unknown = [seed for seed in seeds if seed not in SEEDS]
    if unknown:
        raise ValueError(f"seeds must be among {list(SEEDS)}: {unknown}")
    groups = tuple(args.groups) if args.groups else PHASE2_GROUPS
    unknown_groups = [group for group in groups if group not in PHASE2_GROUPS]
    if unknown_groups:
        raise ValueError(f"groups must be among {list(PHASE2_GROUPS)}: {unknown_groups}")
    if args.phase == 1 and args.groups:
        raise ValueError("phase 1 does not take --groups")
    if args.phase in (3, 4, 5) and args.groups:
        raise ValueError(f"phase {args.phase} does not take --groups")
    args.seeds = seeds
    args.groups = groups
    return args


def history_from_generations(summary: dict, seed: int, epochs_per_generation: int) -> dict[str, object]:
    """Copy one generation-loop summary. Official test is marked at each generation-end epoch."""
    n_test = len(summary["test_acc"])
    n_epochs = len(summary["train_acc"])
    history: dict[str, object] = {
        "seed": seed,
        "generations": n_test,
        "test_acc": [float(value) for value in summary["test_acc"]],
        "test_loss": [float(value) for value in summary.get("test_loss", [])],
        "test_epoch": [],
    }
    if n_test and n_epochs == n_test * epochs_per_generation:
        history["test_epoch"] = [(index + 1) * epochs_per_generation for index in range(n_test)]
    for key in _EPOCH_KEYS:
        history[key] = [float(value) for value in summary[key]]
    for key in _ACTION_KEYS:
        history[key] = list(summary.get(key, []))
    return history


def prefix_clean_loader(loader: DataLoader, count: int) -> DataLoader:
    """Return the first clean images. The slice has no crop, flip, or cutout."""
    dataset = loader.dataset
    count = min(int(count), len(dataset))
    if isinstance(dataset, _CudaCIFAR):
        return _BatchLoader(
            _CudaCIFAR(dataset.images[:count], dataset.labels[:count], augment=False),
            int(loader.batch_size),
            shuffle=True,
        )
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = list(dataset.indices[:count]) if isinstance(dataset, Subset) else list(range(count))
    return DataLoader(Subset(source, indices), batch_size=loader.batch_size, shuffle=True)


def run_seed(
    data: Cifar10Data,
    out_dir: Path,
    seed: int,
    group: str = "seq_conv",
    simulation: tuple[float, int, int] | None = None,
    simulation_alg=None,
    augmented_simulation: bool = False,
    rollout_depth_value: int | None = None,
    clean_prefix: int | None = None,
) -> None:
    """Train one seed with the generation loop. Official test is logged and does not stop the run."""
    if (out_dir / "history.pt").is_file():
        print(f"SKIP seed {seed} history exists at {out_dir / 'history.pt'}", flush=True)
        return
    hp = added_hyperparameters()
    hp["generations"] = GENERATIONS
    hp["epochs"] = EPOCHS
    hp["model_channels"] = CHANNELS
    hp["model_hidden_dim"] = HIDDEN
    if simulation is not None:
        hp["simulation_time"] = float(simulation[0])
        hp["simulation_epochs"] = int(simulation[1])
        hp["simulation_set_size"] = int(simulation[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2 if device.type == "cuda" else 4)
    seed_all(seed)
    model = build_model("added", hp)
    train_loader, val_loader, clean_loader = data.loaders(int(hp["batch_size"]))
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(int(hp["batch_size"]))
    if clean_prefix is not None:
        sim_train, sim_val = prefix_clean_loader(clean_loader, int(clean_prefix)), val_loader
    elif augmented_simulation:
        sim_train, sim_val = train_loader, val_loader
    else:
        sim_train, sim_val = sample_loaders(clean_loader, val_loader, int(hp["simulation_set_size"]), seed=seed)
    cfg = running_config_for_group("added", device)
    cfg.generations = GENERATIONS
    cfg.epochs = EPOCHS
    cfg.device = str(device)
    cfg.lr_scheduler = CosineDecayScheduler(GENERATIONS * EPOCHS)
    cfg.simulation_alg = simulation_alg or montecarlo_alg
    cfg.simulation_score = SimulationScore(weight_acc=1.0, weight_countW=0.0)
    cfg.stopper = AccuracyStopper(target_accuracy=0.999)
    cfg.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    apply_action_group(cfg, group)
    if simulation is not None:
        apply_simulation_budget(cfg, simulation)
    if clean_prefix is not None:
        cfg.simulation_set_size = len(sim_train.dataset)
    elif augmented_simulation:
        cfg.simulation_set_size = len(train_loader.dataset)
    print(
        f"EXP008 seed {seed} group {group} device {device} generations {GENERATIONS} epochs {EPOCHS} "
        f"channels {CHANNELS} hidden {HIDDEN} smooth {LABEL_SMOOTHING} cutout True "
        f"sim {cfg.simulation_scheduler.simulation_time}s "
        f"{cfg.simulation_scheduler.simulation_epochs} epochs set {cfg.simulation_set_size} "
        f"rollout depth {project_config.MCTS_ROLLOUT_DEPTH if rollout_depth_value is None else rollout_depth_value}",
        flush=True,
    )
    with rollout_depth(rollout_depth_value):
        grown, summary = train_generations(
            model, train_loader, val_loader, cfg,
            sim_train_loader=sim_train, sim_val_loader=sim_val, test_loader=test_loader,
        )
    print(f"TEST seed {seed}", [round(float(value), 4) for value in summary["test_acc"]], flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = history_from_generations(summary, seed, EPOCHS)
    history["action_group"] = group
    history["simulation_time"] = float(cfg.simulation_scheduler.simulation_time)
    history["simulation_epochs"] = int(cfg.simulation_scheduler.simulation_epochs)
    history["simulation_set_size"] = int(cfg.simulation_set_size)
    history["simulation_alg"] = cfg.simulation_alg.__name__.rsplit(".", 1)[-1]
    history["simulation_augmented"] = bool(augmented_simulation and clean_prefix is None)
    history["simulation_clean_prefix"] = 0 if clean_prefix is None else int(clean_prefix)
    history["simulation_rollout_depth"] = (
        project_config.MCTS_ROLLOUT_DEPTH if rollout_depth_value is None else int(rollout_depth_value)
    )
    torch.save(history, out_dir / "history.pt")
    torch.save(
        {"state_dict": grown.state_dict(), "test_acc": summary["test_acc"], "seed": seed},
        out_dir / "grown.pt",
    )
    print(f"HISTORY -> {out_dir / 'history.pt'} best_test {max(summary['test_acc']):.4f}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Run one experiment phase. Phase 5 grades the first 5,000 clean images."""
    args = parse_args(argv)
    data = Cifar10Data(DATA_DIR)
    data.prepare()
    if args.phase == 1:
        print(f"Exp 008 phase 1 seeds {', '.join(str(seed) for seed in args.seeds)}.", flush=True)
        print(f"Write target: {RUNS_DIR}", flush=True)
        for seed in args.seeds:
            run_seed(data, RUNS_DIR / f"seed_{seed}", seed, "seq_conv")
        return
    if args.phase == 3:
        time_s, sim_epochs, set_size = PHASE3_SIMULATION
        print(
            f"Exp 008 phase 3 all seeds {', '.join(str(seed) for seed in args.seeds)} "
            f"sim {time_s}s {sim_epochs} epochs set {set_size}.",
            flush=True,
        )
        print(f"Write target: {PHASE3_DIR}", flush=True)
        for seed in args.seeds:
            run_seed(data, PHASE3_DIR / "all" / f"seed_{seed}", seed, "all", PHASE3_SIMULATION)
        return
    if args.phase == 4:
        epoch_list = ", ".join(str(simulation[1]) for simulation in PHASE4_SIMULATIONS)
        print(
            f"Exp 008 phase 4 all seeds {', '.join(str(seed) for seed in args.seeds)} "
            f"sim {PHASE4_SIMULATIONS[0][0]}s epochs {epoch_list} "
            f"augmented set {PHASE4_SIMULATIONS[0][2]} rollout depth {PHASE4_ROLLOUT_DEPTH}.",
            flush=True,
        )
        print(f"Write target: {PHASE4_DIR}", flush=True)
        for simulation in PHASE4_SIMULATIONS:
            sim_epochs = simulation[1]
            for seed in args.seeds:
                run_seed(
                    data,
                    PHASE4_DIR / f"epochs_{sim_epochs}" / f"seed_{seed}",
                    seed,
                    "all",
                    simulation,
                    augmented_simulation=True,
                    rollout_depth_value=PHASE4_ROLLOUT_DEPTH,
                )
        return
    if args.phase == 5:
        epoch_list = ", ".join(str(simulation[1]) for simulation in PHASE5_SIMULATIONS)
        print(
            f"Exp 008 phase 5 all seeds {', '.join(str(seed) for seed in args.seeds)} "
            f"sim {PHASE5_SIMULATIONS[0][0]}s epochs {epoch_list} "
            f"clean prefix {PHASE5_CLEAN_PREFIX} rollout depth {PHASE5_ROLLOUT_DEPTH}.",
            flush=True,
        )
        print(f"Write target: {PHASE5_DIR}", flush=True)
        for simulation in PHASE5_SIMULATIONS:
            sim_epochs = simulation[1]
            for seed in args.seeds:
                run_seed(
                    data,
                    PHASE5_DIR / f"epochs_{sim_epochs}" / f"seed_{seed}",
                    seed,
                    "all",
                    simulation,
                    rollout_depth_value=PHASE5_ROLLOUT_DEPTH,
                    clean_prefix=PHASE5_CLEAN_PREFIX,
                )
        return
    print(
        f"Exp 008 phase 2 groups {', '.join(args.groups)} seeds {', '.join(str(seed) for seed in args.seeds)}.",
        flush=True,
    )
    print(f"Write target: {PHASE2_DIR}", flush=True)
    for group in args.groups:
        for seed in args.seeds:
            run_seed(data, PHASE2_DIR / group / f"seed_{seed}", seed, group)


if __name__ == "__main__":
    main()
