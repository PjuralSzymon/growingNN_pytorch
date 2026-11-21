"""Simulation-set selectors. Each algorithm lives in its own module."""

from growingnn.simulation.simulation_sets.base import SimulationSetAlgorithm
from growingnn.simulation.simulation_sets.ccs import CcsSimulationSet
from growingnn.simulation.simulation_sets.el2n import El2nSimulationSet
from growingnn.simulation.simulation_sets.moderate import ModerateSimulationSet
from growingnn.simulation.simulation_sets.stratified_random import StratifiedRandomSimulationSet

SIMULATION_SET_ALGORITHMS: tuple[type[SimulationSetAlgorithm], ...] = (
    StratifiedRandomSimulationSet,
    El2nSimulationSet,
    CcsSimulationSet,
    ModerateSimulationSet,
)

__all__ = [
    "SIMULATION_SET_ALGORITHMS",
    "CcsSimulationSet",
    "El2nSimulationSet",
    "ModerateSimulationSet",
    "SimulationSetAlgorithm",
    "StratifiedRandomSimulationSet",
]
