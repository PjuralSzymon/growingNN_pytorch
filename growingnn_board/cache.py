"""In-memory cache updated by the polling watcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from growingnn_board.file_reader import read_json, read_main, read_training_metrics


@dataclass
class ExperimentCache:
    path: Path | None = None
    main: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    generations: dict[int, dict[str, Any]] = field(default_factory=dict)
    simulations: dict[int, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def load(self, experiment_path: Path) -> None:
        warnings: list[str] = []
        main_path = experiment_path / "main.json"
        parsed = read_main(main_path)
        if parsed is None:
            if self.path == experiment_path and self.main is not None:
                warnings.append("main.json temporarily invalid; keeping last valid data")
                self.warnings = warnings
                return
            self.path = experiment_path
            self.main = None
            self.warnings = ["main.json missing or invalid"]
            return

        main = parsed.model_dump()
        training_path = experiment_path / "metrics" / "training.json"
        training = read_training_metrics(training_path)
        generations: dict[int, dict[str, Any]] = {}
        gen_dir = experiment_path / "generations"
        if gen_dir.is_dir():
            for file in gen_dir.glob("generation_*.json"):
                data = read_json(file)
                if data and "generation" in data:
                    generations[int(data["generation"])] = data

        simulations: dict[int, dict[str, Any]] = {}
        sim_dir = experiment_path / "simulations"
        if sim_dir.is_dir():
            for file in sim_dir.glob("simulation_gen_*.json"):
                data = read_json(file)
                if data and "generation" in data:
                    simulations[int(data["generation"])] = data

        self.path = experiment_path
        self.main = main
        self.training = training.model_dump() if training else self.training
        self.generations = generations or self.generations
        self.simulations = simulations or self.simulations
        self.warnings = warnings
