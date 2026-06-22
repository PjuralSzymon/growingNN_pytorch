"""Pydantic schemas for GrowingNN Board JSON files."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainingParameters(BaseModel):
    totalGenerations: int = 0
    epochsPerGeneration: int = 0
    currentGeneration: int = 0
    currentEpoch: int = 0
    totalEpochs: int = 0
    completedGlobalEpochs: int = 0
    simulationAlgorithm: str = ""
    simulationTimeSec: float = 0.0
    learningRateAlpha: float | None = None


class MainExperiment(BaseModel):
    experimentId: str
    experimentName: str
    lastUpdate: str
    experimentStartedOn: str
    trainingTimeElapsedSec: int = 0
    status: str = "running"
    dataset: str = ""
    device: str = ""
    trainingParameters: TrainingParameters = Field(default_factory=TrainingParameters)
    generationTimeline: list[dict[str, Any]] = Field(default_factory=list)
    lastSimulation: dict[str, Any] | None = None
    graphs: dict[str, str] = Field(default_factory=dict)


class TrainingMetrics(BaseModel):
    lastUpdate: str
    epochs: list[dict[str, Any]] = Field(default_factory=list)
