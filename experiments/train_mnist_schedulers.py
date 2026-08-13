"""Run the train_mnist grid once for every simulation scheduler."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import torch

from experiments import experiments_common as common
from experiments import train_mnist
from growingnn.simulation.simulation_schedulers import (
    AlwaysSimulationScheduler,
    MeanStandardDeviationStagnationSimulationScheduler,
    NeverSimulationScheduler,
    ProgressCheckSimulationScheduler,
    SlopeEstimationSimulationScheduler,
)

RUNS_DIR = train_mnist.RUNS_DIR / "scheduler_experiment"
SCHEDULERS = (
    AlwaysSimulationScheduler,
    ProgressCheckSimulationScheduler,
    SlopeEstimationSimulationScheduler,
    MeanStandardDeviationStagnationSimulationScheduler,
    NeverSimulationScheduler,
)


if __name__ == "__main__":
    args = common.parse_board_cli("Compare simulation schedulers on train_mnist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train_mnist.MNISTData(train_mnist.DATA_DIR)
    data.prepare()

    for scheduler in SCHEDULERS:
        scheduler_name = scheduler.mode.name.lower()
        definition = common.ExperimentDefinition(
            name=f"MNIST {scheduler_name}",
            runs_dir=RUNS_DIR / scheduler_name,
            history_filename=train_mnist.MNIST_HISTORY_FILENAME,
            seeds=train_mnist.GRID_SEEDS,
            folder_name=train_mnist.build_mnist_hyperparameter_folder_name,
            model_factory=train_mnist._build_model,
            loader_factory=lambda hp: data.loaders(int(hp["batch_size"])),
            board_metadata=lambda hp, folder, seed: (
                f"MNIST {scheduler_name} | {folder} | seed {seed}",
                "MNIST",
            ),
        )
        with patch.object(common, "AlwaysSimulationScheduler", scheduler):
            executed, skipped = common.run_experiment_grid(
                definition,
                (
                    dict(zip(train_mnist.METAPARAM_KEYS, values))
                    for values in itertools.product(*train_mnist.METAPARAM_LISTS)
                ),
                device=device,
                board=args.board,
            )
        print(
            f"{scheduler_name}: executed {executed}, skipped {skipped}, "
            f"output {definition.runs_dir}"
        )
