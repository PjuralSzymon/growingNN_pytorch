"""Experiment 010: sequential two-pass tuning of MCTS meta-parameters.

Runs CIFAR-10 with a very small MNIST-style starter graph. Each candidate is
evaluated on seeds 1 and 2. Within each pass,
parameters are tuned in order and the selected value carries forward. The
completed first-pass choices initialize the second pass.

Run all phases with:
    pipenv run python -u experiments/train_cifar10_exp010_mcts_meta.py

The run is long. Use --pass 1 or --pass 2 to limit execution. State and
per-candidate histories are persisted so interrupted work can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config as project_config
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_cifar10_exp009_best_reference import get_datasets
from growingnn.core.config import RunningConfig
from growingnn.board.experiment_board import ExperimentBoard
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.lr_scheduler_action import ActionLearningRateScheduler, ScheduleMode
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

# Ten generations with twenty real-training epochs per generation.
GENERATIONS = 10
EPOCHS = 20

RUNS_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp010_mcts_meta_g10_tiny"
STATE_PATH = RUNS_DIR / "experiment_state.json"
DATASETS = ("cifar10",)
SEEDS = (1, 2)
PHASE3_SEEDS = (1, 2, 3)
PHASE3_GENERATIONS = 40


class TinyCifar10Starter(nn.Module):
    """MNIST-style CIFAR starter: one 8-channel convolution and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.linear2 = nn.Linear(8, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


def initial_graph(dataset: str = "cifar10") -> nn.Module:
    if dataset != "cifar10":
        raise ValueError(f"Experiment 010 only supports CIFAR-10, got {dataset!r}")
    return TinyCifar10Starter()


PARAMETER_ORDER = (
    "MCTS_UCB1_C",
    "MCTS_UCB1_USE_SQRT",
    "MCTS_PROPAGATE_ROLLOUT_VALUE",
    "MCTS_ROLLOUT_DEPTH",
    "MCTS_ROLLOUT_EPOCHS",
    "MCTS_ROLLOUT_LR",
    "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS",
    "simulation_time",
    "simulation_set_percent",
)
BASE_CANDIDATES: dict[str, tuple[Any, ...]] = {
    "MCTS_UCB1_C": (0.5, 1.0, 2.0, 4.0, 8.0),
    "MCTS_UCB1_USE_SQRT": (False, True),
    "MCTS_PROPAGATE_ROLLOUT_VALUE": (False, True),
    "MCTS_ROLLOUT_DEPTH": (0, 1, 2, 3),
    "MCTS_ROLLOUT_EPOCHS": (1, 2, 4),
    # 0.0001 is the current configured value; the scheduler floors applied LR at 0.001.
    # Avoid 0.001 as a separate arm because it is behaviorally identical to 0.0001.
    "MCTS_ROLLOUT_LR": (0.0001, 0.003, 0.01),
    "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS": (1, 3, 5),
    "simulation_time": (8.0, 16.0, 32.0),
    # The first value is replaced with the exact percentage equivalent of 128.
    "simulation_set_percent": (5.0, 10.0, 30.0, 40.0),
}
BASELINE: dict[str, Any] = {
    "MCTS_UCB1_C": 2.0,
    "MCTS_UCB1_USE_SQRT": False,
    "MCTS_PROPAGATE_ROLLOUT_VALUE": False,
    "MCTS_ROLLOUT_DEPTH": 2,
    "MCTS_ROLLOUT_EPOCHS": 1,
    "MCTS_ROLLOUT_LR": 0.0001,
    "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS": 3,
    "simulation_time": 8.0,
    "simulation_set_percent": None,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {"type": type(value).__name__, **value.__dict__}
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def current_mcts_values() -> dict[str, Any]:
    """Capture process globals so every candidate can start from a known state."""
    return {
        "MCTS_UCB1_C": project_config.MCTS_UCB1_C,
        "MCTS_UCB1_USE_SQRT": project_config.MCTS_UCB1_USE_SQRT,
        "MCTS_PROPAGATE_ROLLOUT_VALUE": project_config.MCTS_PROPAGATE_ROLLOUT_VALUE,
        "MCTS_ROLLOUT_DEPTH": project_config.MCTS_ROLLOUT_DEPTH,
        "MCTS_ROLLOUT_EPOCHS": project_config.MCTS_ROLLOUT_EPOCHS,
        "MCTS_ROLLOUT_LR": project_config.MCTS_ROLLOUT_LR,
        "SIMULATION_MIN_ALGORITHM_ITERATION_RUNS": project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS,
    }


def apply_values(values: dict[str, Any]) -> None:
    for name in PARAMETER_ORDER:
        if name not in values or name in ("simulation_time", "simulation_set_percent"):
            continue
        value = values[name]
        if name == "MCTS_ROLLOUT_LR" and isinstance(value, (int, float)):
            value = ActionLearningRateScheduler(ScheduleMode.CONSTANT, alpha=float(value))
        setattr(project_config, name, value)


def initial_state() -> dict[str, Any]:
    defaults = dict(BASELINE)
    defaults["MCTS_ROLLOUT_LR"] = 0.0001
    return {
        "experiment": "010_mcts_meta",
        "status": "running",
        "baseline": "Experiment 009 CIFAR-10 recipe; seeds 1 and 2",
        "pass": 1,
        "parameter_index": 0,
        "selected_values": defaults,
        "pass_winners": {"1": {}, "2": {}},
        "completed_candidates": {},
        "selection_metric": "mean final validation accuracy across CIFAR-10 seeds 1 and 2",
    }


def get_state(reset: bool = False) -> dict[str, Any]:
    if reset or not STATE_PATH.exists():
        state = initial_state()
        _write_json(STATE_PATH, state)
        return state
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def candidate_values(parameter: str, data) -> tuple[Any, ...]:
    values = list(BASE_CANDIDATES[parameter])
    if parameter == "simulation_set_percent":
        clean_train_size = len(data.loaders(128)[2].dataset)
        exact_default_percent = 100.0 * 128 / clean_train_size
        values.insert(0, exact_default_percent)
    return tuple(values)


def run_candidate(
    dataset: str,
    seed: int,
    parameter: str,
    value: Any,
    values: dict[str, Any],
    data,
    output_dir: Path,
    generations: int = GENERATIONS,
    phase_name: str = "pass",
) -> dict[str, Any]:
    history_path = output_dir / "history.pt"
    metadata_path = output_dir / "run.json"
    if history_path.is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    active = dict(values)
    active[parameter] = value
    apply_values(active)
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initial_graph(dataset)
    train_loader, val_loader, clean_loader = data.loaders(128)
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(128)

    if parameter == "simulation_set_percent" or active.get("simulation_set_percent") is not None:
        percent = float(active.get("simulation_set_percent") or 100.0 * 128 / len(clean_loader.dataset))
        requested_size = max(1, min(len(clean_loader.dataset), round(len(clean_loader.dataset) * percent / 100)))
    else:
        percent = 100.0 * 128 / len(clean_loader.dataset)
        requested_size = 128
    sim_train, sim_val = sample_loaders(clean_loader, val_loader, requested_size, seed=seed)
    actual_sim_train_size = len(sim_train.dataset)
    actual_sim_val_size = len(sim_val.dataset)
    board = ExperimentBoard(
        output_dir / "board",
        experiment_id=f"exp010_{parameter}_{str(value).replace('.', 'p')}_{dataset}_seed{seed}",
        experiment_name="Experiment 010 MCTS meta-parameter tuning",
        dataset=dataset,
        device=str(device),
    )

    config = RunningConfig(
        generations=generations,
        epochs=EPOCHS,
        lr_scheduler=CosineDecayScheduler(total_epochs=generations * EPOCHS),
        stopper=AccuracyStopper(target_accuracy=0.999),
        simulation_alg=montecarlo_alg,
        simulation_scheduler=AlwaysSimulationScheduler(float(active["simulation_time"]), 1),
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
        simulation_set_size=actual_sim_train_size,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        device=device,
        momentum=0.9,
        weight_decay=0.0001,
        max_grad_norm=1.0,
        seq_conv_insert_before_global_pool=True,
        enable_experiment_board=True,
        experiment_board=board,
    )

    print(
        f"EXP010 {phase_name}={output_dir.parts[-5]} parameter={parameter} value={value} "
        f"dataset={dataset} seed={seed} sim_train={actual_sim_train_size}/{len(clean_loader.dataset)} "
        f"({percent:.4f}%) sim_val={actual_sim_val_size} time={active['simulation_time']}s",
        flush=True,
    )
    _grown, summary = train_generations(
        model,
        train_loader,
        val_loader,
        config,
        sim_train_loader=sim_train,
        sim_val_loader=sim_val,
        test_loader=test_loader,
    )
    history = {
        "experiment": "010_mcts_meta",
        "dataset": dataset,
        "seed": seed,
        "parameter": parameter,
        "candidate_value": value,
        "phase": phase_name,
        "generations": generations,
        "epochs_per_generation": EPOCHS,
        "total_epochs": generations * EPOCHS,
        "active_values": active,
        "simulation_set_percent_requested": percent,
        "simulation_train_size_requested": requested_size,
        "simulation_train_size_actual": actual_sim_train_size,
        "simulation_val_size_actual": actual_sim_val_size,
        **{key: list(summary[key]) for key in summary},
    }
    torch.save(history, history_path)
    result = {
        "dataset": dataset,
        "seed": seed,
        "parameter": parameter,
        "candidate_value": value,
        "final_train_accuracy": float(summary["train_acc"][-1]),
        "final_validation_accuracy": float(summary["val_acc"][-1]),
        "best_validation_accuracy": float(max(summary["val_acc"])),
        "final_test_accuracy": float(summary["test_acc"][-1]),
        "best_test_accuracy": float(max(summary["test_acc"])),
        "simulation_set_percent_requested": percent,
        "simulation_train_size_actual": actual_sim_train_size,
        "simulation_val_size_actual": actual_sim_val_size,
        "history": str(history_path),
    }
    _write_json(metadata_path, result)
    print(f"RESULT {json.dumps(result)}", flush=True)
    return result


def candidate_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def run_experiment(selected_passes: tuple[int, ...], reset: bool = False) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    state = get_state(reset=reset)
    data_by_dataset = {name: get_datasets(name) for name in DATASETS}

    for pass_number in selected_passes:
        if pass_number == 2 and any(parameter not in state["pass_winners"]["1"] for parameter in PARAMETER_ORDER):
            raise RuntimeError("Pass 2 requires completed winners for every parameter in pass 1")
        state["pass"] = pass_number
        values = dict(BASELINE if pass_number == 1 else state["pass_winners"]["1"])
        if pass_number == 1:
            values.update(current_mcts_values())
            values["simulation_time"] = 8.0
            values["simulation_set_percent"] = None
        apply_values(values)

        for parameter_index, parameter in enumerate(PARAMETER_ORDER):
            state["parameter_index"] = parameter_index
            candidate_results = []
            for candidate in candidate_values(parameter, data_by_dataset["cifar10"]):
                key = f"pass_{pass_number}/{parameter}/{candidate_key(candidate)}"
                if key not in state["completed_candidates"]:
                    all_runs = []
                    for dataset in DATASETS:
                        for seed in SEEDS:
                            candidate_dir = RUNS_DIR / f"pass_{pass_number}" / parameter / f"value_{str(candidate).replace('.', 'p')}" / dataset / f"seed_{seed}"
                            result = run_candidate(
                                dataset,
                                seed,
                                parameter,
                                candidate,
                                values,
                                data_by_dataset[dataset],
                                candidate_dir,
                            )
                            all_runs.append(result)
                    state["completed_candidates"][key] = all_runs
                    _write_json(STATE_PATH, state)
                candidate_results.append(state["completed_candidates"][key])

            def score(runs: list[dict[str, Any]]) -> float:
                by_dataset = {
                    dataset: [row["final_validation_accuracy"] for row in runs if row["dataset"] == dataset]
                    for dataset in DATASETS
                }
                return sum(sum(scores) / len(scores) for scores in by_dataset.values()) / len(DATASETS)

            candidate_scores = [
                {"value": value, "score": score(runs), "runs": runs}
                for value, runs in zip(candidate_values(parameter, data_by_dataset["cifar10"]), candidate_results)
            ]
            winner = max(candidate_scores, key=lambda row: row["score"])
            values[parameter] = winner["value"]
            state["selected_values"] = values
            state["pass_winners"][str(pass_number)][parameter] = winner["value"]
            state.setdefault("parameter_comparisons", {}).setdefault(str(pass_number), {})[parameter] = candidate_scores
            _write_json(STATE_PATH, state)
            _write_json(RUNS_DIR / f"pass_{pass_number}" / "selected_values.json", values)
            print(f"WINNER pass={pass_number} parameter={parameter} value={winner['value']} mean_final_val={winner['score']:.6f}", flush=True)

        _write_json(RUNS_DIR / f"pass_{pass_number}" / "selected_values.json", values)

    if all(str(p) in state["pass_winners"] and len(state["pass_winners"][str(p)]) == len(PARAMETER_ORDER) for p in (1, 2)):
        state["status"] = "complete"
        _write_json(RUNS_DIR / "final_selected_values.json", state["pass_winners"]["2"])
    _write_json(STATE_PATH, state)


def run_phase3() -> None:
    """Repeat the final Pass 2 configuration for three 40-generation runs."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    final_values_path = RUNS_DIR / "final_selected_values.json"
    if not final_values_path.is_file():
        raise RuntimeError("Phase 3 requires completed Pass 2 winners")
    values = json.loads(final_values_path.read_text(encoding="utf-8"))
    if any(parameter not in values for parameter in PARAMETER_ORDER):
        raise RuntimeError("Phase 3 final settings do not contain every tuned parameter")

    phase_dir = RUNS_DIR / "phase_3"
    state_path = phase_dir / "phase_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "experiment": "010_mcts_meta",
        "phase": 3,
        "status": "running",
        "dataset": "cifar10",
        "seeds": list(PHASE3_SEEDS),
        "generations": PHASE3_GENERATIONS,
        "epochs_per_generation": EPOCHS,
        "total_epochs_per_run": PHASE3_GENERATIONS * EPOCHS,
        "selected_values": values,
        "runs": {},
    }
    state["status"] = "running"
    state["selected_values"] = values
    _write_json(state_path, state)
    data = get_datasets("cifar10")
    apply_values(values)

    for seed in PHASE3_SEEDS:
        key = f"cifar10/seed_{seed}"
        run_dir = phase_dir / "final_best" / "cifar10" / f"seed_{seed}"
        result = run_candidate(
            "cifar10",
            seed,
            "phase3_final_configuration",
            "pass2_winners",
            values,
            data,
            run_dir,
            generations=PHASE3_GENERATIONS,
            phase_name="phase3",
        )
        state["runs"][key] = result
        _write_json(state_path, state)

    state["status"] = "complete"
    _write_json(state_path, state)
    print(f"PHASE3 COMPLETE seeds={PHASE3_SEEDS} selected_values={values}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass", dest="passes", choices=("1", "2", "all"), default="all")
    parser.add_argument("--phase3", action="store_true", help="Run the final settings three times for 40 generations")
    parser.add_argument("--reset", action="store_true", help="Start a fresh selection state; completed run folders are reused.")
    args = parser.parse_args()
    if args.phase3:
        run_phase3()
        return
    passes = (1, 2) if args.passes == "all" else (int(args.passes),)
    run_experiment(passes, reset=args.reset)


if __name__ == "__main__":
    main()
