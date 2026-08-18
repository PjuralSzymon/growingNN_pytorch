"""
Experiment 005 driver — compare simulation search algorithms on the Exp 004 best LR package.

Fixed package from Experiment 004 conclusions:
- LR: composed_exponential × logistic recovery
- simulation grading: validation accuracy
- slope gate: 3°
- recovery warmup: logistic, warmup_iterations=10, k=10
- epochs per generation: 10
- generations: 10
- simulation time: 120 s
- five matched seeds per simulation algorithm (same idea as Exp 004's matched seeds per LR schedule)

Grid factors:
1. simulation algorithm (MCTS, greedy, random, and the new candidate algs)
2. starter architecture (big and medium, same Exp 003 pair)

Each algorithm × starter runs on SEEDS = (100, 101, 102, 103, 104).

Why medium is included:
The big starter may need only one strong action to reach a high plateau, which can make
greedy look artificially strong. Medium (1×Conv + 2×Linear) should need more growth steps,
so lookahead and multi-step search have a clearer job.

Simulation algorithm IDs (also the first folder under RUNS_DIR):

| ID | Module |
| --- | --- |
| montecarlo | montecarlo_alg |
| greedy | greedy_alg |
| random | random_alg |
| sequential_halving | sequential_halving_alg |
| ugape | ugape_alg |
| successive_rejects | successive_rejects_alg |
| beam_search | beam_search_alg |
| best_first | best_first_alg |
| shot | shot_alg |
| sequential_halving_beam | sequential_halving_beam_alg |
| ugape_deepen | ugape_deepen_alg |
| progressive_widening | progressive_widening_alg |
| hierarchical_search | hierarchical_search_alg |

Example run path:
experiments/output/train_mnist/runs/exp005_simulation_algorithms/<alg_id>/<model_name>/<hp_folder>/seed_<seed>/
"""

from __future__ import annotations

import itertools
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import experiments_common as common
from experiments import train_mnist
from experiments.train_mnist_exp001_slope_model_depth import configure_deterministic_seeding
from experiments.train_mnist_exp002_initial_architectures import (
    CHANNELS,
    HIDDEN_LINEAR_SIZE,
    BigAvgPoolMnistNet,
    Medium1Conv2LinearMnistNet,
)
from experiments.train_mnist_exp004_composed_lr_schedulers import (
    EPOCHS_PER_GENERATION,
    GENERATIONS,
    INITIAL_LR,
    SCORE_ACCURACY_METRIC,
    SIMULATION_TIME_SEC,
    SLOPE_ANGLE_THRESHOLD,
    WARMUP_ITERATIONS,
    WARMUP_STEEPNESS,
    build_learning_rate_scheduler_for_schedule_id,
)
from growingnn.simulation.simulation_schedulers import SlopeEstimationSimulationScheduler
import growingnn.simulation.simulation_algorithms.beam_search_alg as beam_search_alg
import growingnn.simulation.simulation_algorithms.best_first_alg as best_first_alg
import growingnn.simulation.simulation_algorithms.greedy_alg as greedy_alg
import growingnn.simulation.simulation_algorithms.hierarchical_search_alg as hierarchical_search_alg
import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
import growingnn.simulation.simulation_algorithms.progressive_widening_alg as progressive_widening_alg
import growingnn.simulation.simulation_algorithms.random_alg as random_alg
import growingnn.simulation.simulation_algorithms.sequential_halving_alg as sequential_halving_alg
import growingnn.simulation.simulation_algorithms.sequential_halving_beam_alg as sequential_halving_beam_alg
import growingnn.simulation.simulation_algorithms.shot_alg as shot_alg
import growingnn.simulation.simulation_algorithms.successive_rejects_alg as successive_rejects_alg
import growingnn.simulation.simulation_algorithms.ugape_alg as ugape_alg
import growingnn.simulation.simulation_algorithms.ugape_deepen_alg as ugape_deepen_alg

RUNS_DIR = train_mnist.RUNS_DIR / "exp005_simulation_algorithms"
SCHEDULE_ID = "composed_exponential"

# Matched seeds across every simulation algorithm × starter.
SEED_BASE = 100
SEED_COUNT = 5
SEEDS = tuple(SEED_BASE + offset for offset in range(SEED_COUNT))

ALG_VARIANTS: tuple[tuple[str, object], ...] = (
    ("montecarlo", montecarlo_alg),
    ("greedy", greedy_alg),
    ("random", random_alg),
    ("sequential_halving", sequential_halving_alg),
    ("ugape", ugape_alg),
    ("successive_rejects", successive_rejects_alg),
    ("beam_search", beam_search_alg),
    ("best_first", best_first_alg),
    ("shot", shot_alg),
    ("sequential_halving_beam", sequential_halving_beam_alg),
    ("ugape_deepen", ugape_deepen_alg),
    ("progressive_widening", progressive_widening_alg),
    ("hierarchical_search", hierarchical_search_alg),
)


def _factory(builder: Callable[..., torch.nn.Module], **kwargs: object) -> Callable[[dict[str, object]], torch.nn.Module]:
    def factory(_hp: dict[str, object]) -> torch.nn.Module:
        return builder(**kwargs)

    return factory


# Same Exp 003 starters: big needs fewer growth steps; medium should need more.
MODEL_VARIANTS: tuple[tuple[str, Callable[[dict[str, object]], torch.nn.Module]], ...] = (
    ("big", _factory(BigAvgPoolMnistNet, channels=CHANNELS, hidden_linear_size=HIDDEN_LINEAR_SIZE)),
    (
        "medium_1conv_2linear",
        _factory(
            Medium1Conv2LinearMnistNet,
            channels=CHANNELS,
            hidden_linear_size=HIDDEN_LINEAR_SIZE,
        ),
    ),
)


def print_simulation_algorithm_ids() -> None:
    """Print the Exp 005 simulation algorithm ID catalog and output layout."""
    print("Exp 005 simulation algorithm IDs:")
    for index, (alg_id, alg_module) in enumerate(ALG_VARIANTS, start=1):
        module_name = getattr(alg_module, "__name__", str(alg_module)).rsplit(".", 1)[-1]
        print(f"  {index:>2}. {alg_id:<24} module={module_name}  runs={RUNS_DIR / alg_id}")
    print("Exp 005 starter architectures:")
    for model_name, _ in MODEL_VARIANTS:
        print(f"  - {model_name}")
    print(
        "Run path pattern: "
        f"{RUNS_DIR}/<simulation_alg_id>/<model_name>/<hp_folder>/seed_<seed>/"
    )


if __name__ == "__main__":
    args = common.parse_board_cli(
        "Experiment 005: MNIST simulation algorithm comparison on Exp 004 best LR package"
    )
    configure_deterministic_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()
    print(f"Exp 005 write target: {RUNS_DIR}")
    print(f"Fixed LR package: {SCHEDULE_ID}")
    print_simulation_algorithm_ids()

    for alg_id, alg_module in ALG_VARIANTS:
        for model_name, model_factory in MODEL_VARIANTS:
            definition = common.ExperimentDefinition(
                name=f"MNIST exp005 {alg_id} {model_name}",
                runs_dir=RUNS_DIR / alg_id / model_name,
                history_filename=train_mnist.MNIST_HISTORY_FILENAME,
                seeds=SEEDS,
                folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
                model_factory=model_factory,
                loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
                board_metadata=lambda hp, folder, seed, aid=alg_id, model=model_name: (
                    f"MNIST exp005 simulation_alg_id={aid} ({SCHEDULE_ID}) {model} | {folder} | seed {seed}",
                    "MNIST",
                ),
            )
            with patch.object(
                common,
                "AlwaysSimulationScheduler",
                partial(
                    SlopeEstimationSimulationScheduler,
                    angle_threshold=SLOPE_ANGLE_THRESHOLD,
                ),
            ):
                executed, skipped = common.run_experiment_grid(
                    definition,
                    (
                        {
                            **dict(zip(train_mnist.METAPARAM_KEYS, values)),
                            "epochs": EPOCHS_PER_GENERATION,
                            "generations": GENERATIONS,
                            "simulation_time": SIMULATION_TIME_SEC,
                            "lr_alpha": INITIAL_LR,
                            "score_accuracy_metric": SCORE_ACCURACY_METRIC,
                            "simulation_alg_id": alg_id,
                            "simulation_alg": alg_module,
                            "model_name": model_name,
                            "lr_scheduler_factory": (
                                lambda hp, sid=SCHEDULE_ID: build_learning_rate_scheduler_for_schedule_id(
                                    sid, hp
                                )
                            ),
                        }
                        for values in itertools.product(*train_mnist.METAPARAM_LISTS)
                    ),
                    device=device,
                    board=args.board,
                )
            print(
                f"simulation_alg_id={alg_id} model={model_name}: executed {executed}, skipped {skipped}, "
                f"seeds={SEEDS}, gens={GENERATIONS}, epochs={EPOCHS_PER_GENERATION}, "
                f"simt={SIMULATION_TIME_SEC}, lr={SCHEDULE_ID}, output {definition.runs_dir}"
            )
