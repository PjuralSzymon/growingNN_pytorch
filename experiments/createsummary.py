"""Collect CIFAR-10 grid run artifacts from disk and write a grid search summary."""

from __future__ import annotations

import argparse
import json
import os
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
EXPERIMENT_OUTPUT_ROOT = _EXPERIMENT_DIR / "output"
HISTORY_FILENAME = "train_cifar10_history.pt"
RUN_LOCK_FILENAME = ".running.lock"

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
_ACTION_SHORT_LABEL_RE = re.compile(r"\(\s*([^:(]+)")
_SIMULATION_GEN_RE = re.compile(r"^simulation_gen_(?P<generation>\d+)\.json$")
_ACTION_TYPE_ALIASES: dict[str, str] = {
    "Add Seq Linear Layer Action": "Add Seq Layer Action",
}

ConfigStats = tuple[float, float, str, Hyperparameters, list["RunResult"]]
ParamSpread = tuple[str, float, object, object]


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


@dataclass(frozen=True)
class ActionExecution:
    run_dir: Path
    generation: int
    action_type: str
    train_acc_before: float | None
    train_acc_after: float | None
    train_acc_delta: float | None


@dataclass(frozen=True)
class ActionAnalysis:
    executions: tuple[ActionExecution, ...]
    runs_with_board: int
    runs_without_board: int


class GridSummaryWriter:
    """Build and write the text report that ranks grid-search runs."""

    def __init__(self, allowed_output_root: Path = EXPERIMENT_OUTPUT_ROOT) -> None:
        self._allowed_output_root = allowed_output_root

    def write(self, results: list[RunResult], path: Path) -> None:
        if not results:
            raise ValueError("No completed runs found to summarize")

        by_folder_name = self._group_by_folder_name(results)
        config_stats = self._build_config_stats(by_folder_name)
        action_analysis = collect_action_analysis(results)
        lines = self._build_summary_lines(results, by_folder_name, config_stats, action_analysis)

        safe_path = self._resolve_path_under_root(path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote grid summary to %s", safe_path)

    def _resolve_path_under_root(self, path: Path) -> Path:
        allowed_root = self._allowed_output_root.resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(f"Path must be inside {allowed_root}") from exc
        return resolved

    @staticmethod
    def _group_by_folder_name(results: list[RunResult]) -> dict[str, list[RunResult]]:
        by_folder_name: dict[str, list[RunResult]] = defaultdict(list)
        for result in results:
            by_folder_name[result.hyperparameter_folder_name].append(result)
        return by_folder_name

    @staticmethod
    def _build_config_stats(by_folder_name: dict[str, list[RunResult]]) -> list[ConfigStats]:
        config_stats: list[ConfigStats] = []
        for folder_name, runs in by_folder_name.items():
            accs = [run.best_val_acc for run in runs]
            mean_acc = statistics.mean(accs)
            std_acc = statistics.pstdev(accs) if len(accs) > 1 else 0.0
            config_stats.append((mean_acc, std_acc, folder_name, runs[0].hyperparameters, runs))
        config_stats.sort(key=lambda item: item[0], reverse=True)
        return config_stats

    @staticmethod
    def _seed_count_note(by_folder_name: dict[str, list[RunResult]]) -> str:
        seed_counts = sorted(len(runs) for runs in by_folder_name.values())
        if seed_counts[0] == seed_counts[-1]:
            return f"{seed_counts[0]} seeds each"
        return f"seeds per config: {seed_counts[0]}-{seed_counts[-1]}"

    @staticmethod
    def _format_hyperparameters(hyperparameters: Hyperparameters) -> str:
        extra = sorted(key for key in hyperparameters if key not in CANONICAL_PARAM_KEYS)
        ordered = tuple(key for key in CANONICAL_PARAM_KEYS if key in hyperparameters) + tuple(extra)
        return ", ".join(f"{key}={hyperparameters[key]}" for key in ordered)

    @classmethod
    def _ranked_config_lines(cls, config_stats: list[ConfigStats]) -> list[str]:
        lines: list[str] = []
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
            lines.append(f"    {cls._format_hyperparameters(hyperparameters)}")
        return lines

    @classmethod
    def _all_hyperparameter_keys(cls, results: list[RunResult]) -> tuple[str, ...]:
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

    @classmethod
    def _varying_param_keys(cls, results: list[RunResult]) -> tuple[str, ...]:
        seen: dict[str, set[object]] = defaultdict(set)
        for result in results:
            for key, value in result.hyperparameters.items():
                seen[key].add(value)
        return tuple(
            key
            for key in cls._all_hyperparameter_keys(results)
            if key in seen and len(seen[key]) > 1
        )

    @classmethod
    def _sensitivity_section(
        cls, results: list[RunResult]
    ) -> tuple[list[str], list[ParamSpread]]:
        lines: list[str] = []
        param_spread: list[ParamSpread] = []
        sensitivity_keys = cls._varying_param_keys(results)
        if not sensitivity_keys:
            lines.append("  (all runs share the same hyperparameter values)")
            return lines, param_spread

        for key in sensitivity_keys:
            grouped: dict[object, list[float]] = defaultdict(list)
            for result in results:
                if key in result.hyperparameters:
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
                param_spread.append((key, best_value_acc - worst_value_acc, best_value, worst_value))
            lines.append("")
        return lines, param_spread

    @staticmethod
    def _tuning_priority_lines(param_spread: list[ParamSpread]) -> list[str]:
        lines = ["Suggested tuning priority (largest val_acc spread across tested values):"]
        if not param_spread:
            lines.append("  (no varying hyperparameters)")
            return lines
        for key, spread, best_value, worst_value in sorted(param_spread, key=lambda item: item[1], reverse=True):
            lines.append(f"  {key}: spread={spread:.4f} (best={best_value}, worst={worst_value})")
        return lines

    @staticmethod
    def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
        if not rows:
            return ["  (no data)"]
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
        rule = "  ".join("-" * widths[index] for index in range(len(headers)))
        lines = [header_line, rule]
        for row in rows:
            lines.append("  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row)))
        return lines

    @classmethod
    def _action_analysis_section(cls, analysis: ActionAnalysis, results: list[RunResult]) -> list[str]:
        if not analysis.executions and analysis.runs_without_board == len(results):
            return [
                "",
                "Action analysis (from board/simulations):",
                "  (no board artifacts found under completed runs)",
            ]

        run_metrics = {result.run_dir.resolve(): load_run_accuracy_metrics(result.run_dir) for result in results}
        actions_per_run: dict[Path, set[str]] = defaultdict(set)
        usage_count: dict[str, int] = defaultdict(int)
        train_deltas: dict[str, list[float]] = defaultdict(list)

        for execution in analysis.executions:
            usage_count[execution.action_type] += 1
            actions_per_run[execution.run_dir.resolve()].add(execution.action_type)
            if execution.train_acc_delta is not None:
                train_deltas[execution.action_type].append(execution.train_acc_delta)

        train_acc_by_action: dict[str, list[float]] = defaultdict(list)
        test_acc_by_action: dict[str, list[float]] = defaultdict(list)
        for run_dir, action_types in actions_per_run.items():
            metrics = run_metrics.get(run_dir)
            if metrics is None:
                continue
            best_train_acc, best_test_acc = metrics
            for action_type in action_types:
                train_acc_by_action[action_type].append(best_train_acc)
                test_acc_by_action[action_type].append(best_test_acc)

        def _sorted_metric_rows(
            values: dict[str, list[float]],
            formatter,
            *,
            descending: bool = True,
        ) -> list[tuple[str, ...]]:
            rows: list[tuple[str, ...]] = []
            for action_type, items in sorted(
                values.items(),
                key=lambda item: statistics.mean(item[1]),
                reverse=descending,
            ):
                rows.append((action_type, formatter(items), str(len(items))))
            return rows

        usage_rows = [
            (action_type, str(count))
            for action_type, count in sorted(usage_count.items(), key=lambda item: (-item[1], item[0]))
        ]
        train_rows = _sorted_metric_rows(
            train_acc_by_action,
            lambda items: f"{statistics.mean(items):.4f}",
        )
        test_rows = _sorted_metric_rows(
            test_acc_by_action,
            lambda items: f"{statistics.mean(items):.4f}",
        )
        delta_rows = _sorted_metric_rows(
            train_deltas,
            lambda items: f"{statistics.mean(items):+.4f}",
            descending=True,
        )

        board_note = (
            f"Board data from {analysis.runs_with_board}/{len(results)} completed runs "
            f"({len(analysis.executions)} action executions)."
        )
        return [
            "",
            "Action analysis (from board/simulations):",
            f"  {board_note}",
            "",
            "1. Action usage count:",
            *cls._format_table(("action_type", "count"), usage_rows),
            "",
            "2. Mean best train accuracy by action type (runs that used the action):",
            *cls._format_table(("action_type", "mean_train_acc", "runs"), train_rows),
            "",
            "3. Mean best test accuracy by action type (runs that used the action):",
            "  (CIFAR-10 test split is logged as val_acc during training.)",
            *cls._format_table(("action_type", "mean_test_acc", "runs"), test_rows),
            "",
            "4. Mean train accuracy change after action execution:",
            "  (delta = first train_acc in next generation minus final train_acc before the action.)",
            *cls._format_table(("action_type", "mean_delta", "executions"), delta_rows),
        ]

    @classmethod
    def _build_summary_lines(
        cls,
        results: list[RunResult],
        by_folder_name: dict[str, list[RunResult]],
        config_stats: list[ConfigStats],
        action_analysis: ActionAnalysis,
    ) -> list[str]:
        best_mean, best_std, best_folder_name, best_hyperparameters, _best_runs = config_stats[0]
        sensitivity_lines, param_spread = cls._sensitivity_section(results)
        return [
            "GrowingNN CIFAR-10 grid search summary",
            "=" * 72,
            f"Total runs: {len(results)} ({len(by_folder_name)} configs, {cls._seed_count_note(by_folder_name)})",
            "",
            "Configs ranked by mean best validation accuracy:",
            *cls._ranked_config_lines(config_stats),
            "",
            "Best configuration (by mean best val_acc):",
            f"  folder: {best_folder_name}",
            f"  mean best val_acc: {best_mean:.4f} (std={best_std:.4f})",
            f"  {cls._format_hyperparameters(best_hyperparameters)}",
            "",
            "Parameter sensitivity (mean best val_acc per value):",
            *sensitivity_lines,
            *cls._tuning_priority_lines(param_spread),
            *cls._action_analysis_section(action_analysis, results),
        ]


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


def _resolve_path_under_root(path: Path, root: Path) -> Path:
    return GridSummaryWriter(root)._resolve_path_under_root(path)


def load_step_history(history_path: Path) -> dict[str, list[float]]:
    data = torch.load(history_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {history_path}, got {type(data).__name__}")
    return data


def action_short_label(action_str: str | None) -> str:
    if not action_str:
        return "—"
    match = _ACTION_SHORT_LABEL_RE.search(action_str)
    return match.group(1).strip() if match else action_str[:48]


def normalize_action_type(action_type: str) -> str:
    return _ACTION_TYPE_ALIASES.get(action_type, action_type)


def action_type_from_simulation(simulation: dict[str, object]) -> str | None:
    candidates = simulation.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("chosen"):
                name = candidate.get("name")
                if isinstance(name, str) and name:
                    return name
                action = candidate.get("action")
                if isinstance(action, str):
                    return action_short_label(action)
    action_chosen = simulation.get("actionChosen")
    if isinstance(action_chosen, str) and action_chosen:
        return action_short_label(action_chosen)
    return None


def _load_board_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _first_train_acc_next_generation(board_dir: Path, generation: int) -> float | None:
    training = _load_board_json(board_dir / "metrics" / "training.json")
    if training is not None:
        epochs = training.get("epochs")
        if isinstance(epochs, list):
            for row in epochs:
                if not isinstance(row, dict):
                    continue
                if row.get("generation") == generation + 1 and row.get("epochInGeneration") == 0:
                    train_acc = row.get("trainAcc")
                    if isinstance(train_acc, (int, float)):
                        return float(train_acc)
    next_generation = _load_board_json(board_dir / "generations" / f"generation_{generation + 1}.json")
    if next_generation is not None:
        train_acc = next_generation.get("finalTrainAcc")
        if isinstance(train_acc, (int, float)):
            return float(train_acc)
    return None


def load_board_action_executions(run_dir: Path) -> list[ActionExecution]:
    board_dir = run_dir / "board"
    simulations_dir = board_dir / "simulations"
    if not simulations_dir.is_dir():
        return []

    executions: list[ActionExecution] = []
    for simulation_path in sorted(simulations_dir.glob("simulation_gen_*.json")):
        match = _SIMULATION_GEN_RE.match(simulation_path.name)
        if match is None:
            continue
        generation = int(match.group("generation"))
        simulation = _load_board_json(simulation_path)
        if simulation is None:
            continue
        action_type = action_type_from_simulation(simulation)
        if action_type is None:
            continue
        action_type = normalize_action_type(action_type)

        generation_snapshot = _load_board_json(board_dir / "generations" / f"generation_{generation}.json")
        train_acc_before = None
        if generation_snapshot is not None:
            value = generation_snapshot.get("finalTrainAcc")
            if isinstance(value, (int, float)):
                train_acc_before = float(value)

        train_acc_after = _first_train_acc_next_generation(board_dir, generation)
        train_acc_delta = None
        if train_acc_before is not None and train_acc_after is not None:
            train_acc_delta = train_acc_after - train_acc_before

        executions.append(
            ActionExecution(
                run_dir=run_dir,
                generation=generation,
                action_type=action_type,
                train_acc_before=train_acc_before,
                train_acc_after=train_acc_after,
                train_acc_delta=train_acc_delta,
            )
        )
    return executions


def load_run_accuracy_metrics(run_dir: Path) -> tuple[float, float] | None:
    history_path = run_dir / HISTORY_FILENAME
    if not history_path.is_file():
        return None
    step_history = load_step_history(history_path)
    train_acc = step_history.get("train_acc")
    val_acc = step_history.get("val_acc")
    if not isinstance(train_acc, list) or not train_acc:
        return None
    if not isinstance(val_acc, list) or not val_acc:
        return None
    return max(float(value) for value in train_acc), max(float(value) for value in val_acc)


def collect_action_analysis(results: list[RunResult]) -> ActionAnalysis:
    executions: list[ActionExecution] = []
    runs_with_board = 0
    for result in results:
        run_executions = load_board_action_executions(result.run_dir)
        if run_executions:
            runs_with_board += 1
        executions.extend(run_executions)
    return ActionAnalysis(
        executions=tuple(executions),
        runs_with_board=runs_with_board,
        runs_without_board=len(results) - runs_with_board,
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

    step_history = load_step_history(history_path)
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


def run_dir_for_seed(runs_root: Path, hyperparameter_folder_name: str, seed: int) -> Path:
    """Path to one grid run: runs_root/<config_folder>/seed_<N>."""
    return runs_root / hyperparameter_folder_name / f"seed_{seed}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _clear_stale_run_lock(run_dir: Path) -> None:
    lock = run_dir / RUN_LOCK_FILENAME
    if not lock.is_file():
        return
    try:
        pid = int(lock.read_text(encoding="utf-8").strip().split()[0])
    except (OSError, ValueError):
        pid = -1
    if not _pid_alive(pid):
        lock.unlink(missing_ok=True)


def try_claim_run(run_dir: Path) -> bool:
    """Atomically claim a run directory for this process; False if another live worker owns it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_run_lock(run_dir)
    lock = run_dir / RUN_LOCK_FILENAME
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_run_claim(run_dir: Path) -> None:
    """Drop the worker lock after a run finishes or fails."""
    (run_dir / RUN_LOCK_FILENAME).unlink(missing_ok=True)


def load_completed_run(
    run_dir: Path,
    *,
    hyperparameters: Hyperparameters,
    hyperparameter_folder_name: str,
    seed: int,
) -> RunResult | None:
    """Load a finished run from disk; log and return None when the folder exists without history."""
    if not run_dir.is_dir():
        return None
    result = load_run_result_from_dir(
        run_dir,
        hyperparameters=hyperparameters,
        hyperparameter_folder_name=hyperparameter_folder_name,
        seed=seed,
    )
    if result is None:
        logger.info(
            "Skipping incomplete run %s seed %s (no history)",
            hyperparameter_folder_name,
            seed,
        )
    return result


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
            result = load_completed_run(
                seed_dir,
                hyperparameters=hyperparameters,
                hyperparameter_folder_name=config_dir.name,
                seed=seed,
            )
            if result is None:
                continue
            results.append(result)
    return results


def write_grid_summary(
    results: list[RunResult],
    path: Path,
    *,
    allowed_output_root: Path = EXPERIMENT_OUTPUT_ROOT,
) -> None:
    GridSummaryWriter(allowed_output_root).write(results, path)


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
    runs_dir = _resolve_path_under_root(args.runs_dir, EXPERIMENT_OUTPUT_ROOT)
    output_path = _resolve_path_under_root(args.output, EXPERIMENT_OUTPUT_ROOT)
    results = collect_run_results(runs_dir)
    write_grid_summary(results, output_path)
    print(f"Summary written to {output_path} ({len(results)} runs)")


if __name__ == "__main__":
    main()
