"""Collect CIFAR-10 grid run artifacts from disk and write a grid search summary."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.core.logger import logger

DEFAULT_RUNS_DIR = _EXPERIMENT_DIR / "output" / "train_cifar10" / "runs"
DEFAULT_SUMMARY_PATH = _EXPERIMENT_DIR / "output" / "train_cifar10" / "grid_search_summary.txt"
HISTORY_FILENAME = "train_cifar10_history.pt"

Hyperparameters: TypeAlias = dict[str, object]

# Hyperparameter names used when building or reading result folder names.
CANONICAL_PARAM_KEYS = (
    "generations",
    "epochs",
    "batch_size",
    "lr_alpha",
    "simulation_time",
    "simulation_epochs",
    "simulation_set_size",
    "target_accuracy",
    "score_weight_acc",
    "score_weight_countw",
    "augmentation_factor",
    "model_channels",
    "model_hidden_dim",
)

# These values are stored as whole numbers in folder names (the rest use decimals).
_INT_PARAM_KEYS = frozenset(
    {
        "generations",
        "epochs",
        "batch_size",
        "simulation_epochs",
        "simulation_set_size",
        "model_channels",
        "model_hidden_dim",
    }
)

# Each result folder is named from the hyperparameters used in that run, for example:
#   g10_ep30_bs64_lr0.01_simt500.0_sime15_simsz2000_tgt0.9_wacc1.0_wcw0.2_augf0.5_ch32_hd256
# This regex reads those short codes back into a hyperparameter dictionary.
_HYPERPARAMETER_FOLDER_NAME_RE = re.compile(
    r"^g(?P<generations>\d+)"
    r"_ep(?P<epochs>\d+)"
    r"_bs(?P<batch_size>\d+)"
    r"_lr(?P<lr_alpha>[\d.]+)"
    r"_simt(?P<simulation_time>[\d.]+)"
    r"_sime(?P<simulation_epochs>\d+)"
    r"_simsz(?P<simulation_set_size>\d+)"
    r"_tgt(?P<target_accuracy>[\d.]+)"
    r"_wacc(?P<score_weight_acc>[\d.]+)"
    r"_wcw(?P<score_weight_countw>[\d.]+)"
    r"_augf?(?P<augmentation_factor>[\d.]*)"
    r"_ch(?P<model_channels>\d+)"
    r"_hd(?P<model_hidden_dim>\d+)$"
)
# Inside each hyperparameter folder, repeated runs use subfolders like seed_0, seed_1, ...
_SEED_DIR_RE = re.compile(r"^seed_(?P<seed>\d+)$")


@dataclass(frozen=True)
class RunResult:
    hyperparameters: Hyperparameters
    hyperparameter_folder_name: str
    seed: int
    run_dir: Path
    best_val_acc: float
    final_val_acc: float
    params_before: int
    params_after: int
    architecture_changed: bool


def build_hyperparameter_folder_name(hyperparameters: Hyperparameters) -> str:
    """Build the runs/ subfolder name that encodes one grid-search configuration."""
    return (
        f"g{hyperparameters['generations']}_ep{hyperparameters['epochs']}_bs{hyperparameters['batch_size']}"
        f"_lr{hyperparameters['lr_alpha']}_simt{hyperparameters['simulation_time']}"
        f"_sime{hyperparameters['simulation_epochs']}"
        f"_simsz{hyperparameters['simulation_set_size']}_tgt{hyperparameters['target_accuracy']}"
        f"_wacc{hyperparameters['score_weight_acc']}_wcw{hyperparameters['score_weight_countw']}"
        f"_augf{hyperparameters['augmentation_factor']}"
        f"_ch{hyperparameters['model_channels']}_hd{hyperparameters['model_hidden_dim']}"
    )


def parse_hyperparameters_from_folder_name(folder_name: str) -> Hyperparameters | None:
    """Read hyperparameters from a result folder name; return None if the name does not match."""
    match = _HYPERPARAMETER_FOLDER_NAME_RE.match(folder_name)
    if match is None:
        return None

    hyperparameters: Hyperparameters = {}
    for key in CANONICAL_PARAM_KEYS:
        raw = match.group(key)
        if raw == "":
            continue
        hyperparameters[key] = int(raw) if key in _INT_PARAM_KEYS else float(raw)
    return hyperparameters


def parse_seed_dir(name: str) -> int | None:
    match = _SEED_DIR_RE.match(name)
    if match is None:
        return None
    return int(match.group("seed"))


def _ordered_hyperparameter_keys(hyperparameters: Hyperparameters) -> tuple[str, ...]:
    extra = sorted(key for key in hyperparameters if key not in CANONICAL_PARAM_KEYS)
    return tuple(key for key in CANONICAL_PARAM_KEYS if key in hyperparameters) + tuple(extra)


def format_hyperparameters(hyperparameters: Hyperparameters) -> str:
    return ", ".join(
        f"{key}={hyperparameters[key]}" for key in _ordered_hyperparameter_keys(hyperparameters)
    )


def _all_hyperparameter_keys(results: list[RunResult]) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for key in CANONICAL_PARAM_KEYS:
        if any(key in result.hyperparameters for result in results):
            keys.append(key)
            seen.add(key)
    for result in results:
        for key in sorted(result.hyperparameters):
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return tuple(keys)


def _varying_param_keys(results: list[RunResult]) -> tuple[str, ...]:
    seen: dict[str, set[object]] = defaultdict(set)
    for result in results:
        for key, value in result.hyperparameters.items():
            seen[key].add(value)
    return tuple(
        key
        for key in _all_hyperparameter_keys(results)
        if key in seen and len(seen[key]) > 1
    )


def load_run_result_from_dir(
    run_dir: Path,
    *,
    hyperparameters: Hyperparameters,
    hyperparameter_folder_name: str,
    seed: int,
) -> RunResult | None:
    history_path = run_dir / HISTORY_FILENAME
    if not history_path.is_file():
        return None

    step_history = torch.load(history_path, map_location="cpu", weights_only=False)
    val_acc = step_history["val_acc"]
    param_count = step_history["param_count"]
    params_before = int(param_count[0])
    params_after = int(param_count[-1])
    return RunResult(
        hyperparameters=hyperparameters,
        hyperparameter_folder_name=hyperparameter_folder_name,
        seed=seed,
        run_dir=run_dir,
        best_val_acc=max(val_acc),
        final_val_acc=val_acc[-1],
        params_before=params_before,
        params_after=params_after,
        architecture_changed=params_after != params_before,
    )


def collect_run_results(runs_dir: Path) -> list[RunResult]:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    results: list[RunResult] = []
    for config_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        hyperparameters = parse_hyperparameters_from_folder_name(config_dir.name)
        if hyperparameters is None:
            logger.warning("Skipping folder with unparseable name %s", config_dir)
            continue

        for seed_dir in sorted(path for path in config_dir.iterdir() if path.is_dir()):
            seed = parse_seed_dir(seed_dir.name)
            if seed is None:
                continue
            result = load_run_result_from_dir(
                seed_dir,
                hyperparameters=hyperparameters,
                hyperparameter_folder_name=config_dir.name,
                seed=seed,
            )
            if result is None:
                logger.info(
                    "Skipping incomplete run %s seed %s (no history)",
                    config_dir.name,
                    seed,
                )
                continue
            results.append(result)
    return results


def write_grid_summary(results: list[RunResult], path: Path) -> None:
    if not results:
        raise ValueError("No completed runs found to summarize")

    by_folder_name: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        by_folder_name[result.hyperparameter_folder_name].append(result)

    config_stats: list[tuple[float, float, str, Hyperparameters, list[RunResult]]] = []
    for folder_name, runs in by_folder_name.items():
        accs = [run.best_val_acc for run in runs]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        config_stats.append((mean_acc, std_acc, folder_name, runs[0].hyperparameters, runs))

    config_stats.sort(key=lambda item: item[0], reverse=True)
    best_mean, best_std, best_folder_name, best_hyperparameters, _best_runs = config_stats[0]
    seed_counts = sorted(len(runs) for runs in by_folder_name.values())
    seed_note = (
        f"{seed_counts[0]} seeds each"
        if seed_counts[0] == seed_counts[-1]
        else f"seeds per config: {seed_counts[0]}-{seed_counts[-1]}"
    )

    lines = [
        "GrowingNN CIFAR-10 grid search summary",
        "=" * 72,
        f"Total runs: {len(results)} ({len(by_folder_name)} configs, {seed_note})",
        "",
        "Configs ranked by mean best validation accuracy:",
    ]
    for rank, (mean_acc, std_acc, folder_name, hyperparameters, runs) in enumerate(
        config_stats, start=1
    ):
        seeds = ", ".join(str(run.seed) for run in sorted(runs, key=lambda run: run.seed))
        acc_list = ", ".join(
            f"{run.best_val_acc:.4f}" for run in sorted(runs, key=lambda run: run.seed)
        )
        lines.append(
            f"{rank:>2}. {folder_name} | mean={mean_acc:.4f} std={std_acc:.4f} | "
            f"seeds=[{seeds}] acc=[{acc_list}]"
        )
        lines.append(f"    {format_hyperparameters(hyperparameters)}")

    lines.extend(
        [
            "",
            "Best configuration (by mean best val_acc):",
            f"  folder: {best_folder_name}",
            f"  mean best val_acc: {best_mean:.4f} (std={best_std:.4f})",
            f"  {format_hyperparameters(best_hyperparameters)}",
            "",
            "Parameter sensitivity (mean best val_acc per value):",
        ]
    )

    param_spread: list[tuple[str, float, object, object]] = []
    sensitivity_keys = _varying_param_keys(results)
    if not sensitivity_keys:
        lines.append("  (all runs share the same hyperparameter values)")
    for key in sensitivity_keys:
        grouped: dict[object, list[float]] = defaultdict(list)
        for result in results:
            if key not in result.hyperparameters:
                continue
            grouped[result.hyperparameters[key]].append(result.best_val_acc)
        lines.append(f"{key}:")
        value_stats = []
        for value, accs in sorted(grouped.items(), key=lambda item: str(item[0])):
            mean_acc = statistics.mean(accs)
            value_stats.append((value, mean_acc))
            lines.append(f"  {value}: mean={mean_acc:.4f} (n={len(accs)})")
        if len(value_stats) > 1:
            best_value, best_value_acc = max(value_stats, key=lambda item: item[1])
            worst_value, worst_value_acc = min(value_stats, key=lambda item: item[1])
            spread = best_value_acc - worst_value_acc
            param_spread.append((key, spread, best_value, worst_value))
        lines.append("")

    lines.append("Suggested tuning priority (largest val_acc spread across tested values):")
    if not param_spread:
        lines.append("  (no varying hyperparameters)")
    for key, spread, best_value, worst_value in sorted(param_spread, key=lambda item: item[1], reverse=True):
        lines.append(
            f"  {key}: spread={spread:.4f} (best={best_value}, worst={worst_value})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote grid summary to %s", path)


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed CIFAR-10 grid runs from experiments/output/train_cifar10/runs"
    )
    parser.add_argument(
        "runs_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=(
            "Directory containing hyperparameter_folder_name/seed_N run folders "
            f"(default: {DEFAULT_RUNS_DIR})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary output path (default: {DEFAULT_SUMMARY_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_cli(argv)
    results = collect_run_results(args.runs_dir)
    write_grid_summary(results, args.output)
    print(f"Summary written to {args.output} ({len(results)} runs)")


if __name__ == "__main__":
    main()
