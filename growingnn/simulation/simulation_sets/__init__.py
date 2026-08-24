"""Simulation-set generators used by train_generations."""

from growingnn.simulation.simulation_sets.base import SimulationSet
from growingnn.simulation.simulation_sets.protected import ProtectedSimulationSet
from growingnn.simulation.simulation_sets.craig import CraigSimulationSet
from growingnn.simulation.simulation_sets.el2n import El2nSimulationSet
from growingnn.simulation.simulation_sets.grad_match import GradMatchSimulationSet
from growingnn.simulation.simulation_sets.grand import GrandSimulationSet
from growingnn.simulation.simulation_sets.hcdc import HcdcSimulationSet
from growingnn.simulation.simulation_sets.kcenter import KCenterSimulationSet
from growingnn.simulation.simulation_sets.model_drift import ModelDriftSimulationSet
from growingnn.simulation.simulation_sets.moderate_difficulty import ModerateDifficultySimulationSet

SIMULATION_SET_REGISTRY = {
    "protected": ProtectedSimulationSet,
    "moderate_difficulty": ModerateDifficultySimulationSet,
    "model_drift": ModelDriftSimulationSet,
    "grad_match": GradMatchSimulationSet,
    "hcdc": HcdcSimulationSet,
    "kcenter": KCenterSimulationSet,
    "grand": GrandSimulationSet,
    "el2n": El2nSimulationSet,
    "craig": CraigSimulationSet,
}


def build_simulation_set(name: str, **kwargs) -> SimulationSet:
    return SIMULATION_SET_REGISTRY[name](**kwargs)
