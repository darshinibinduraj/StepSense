"""Shared data schemas for StepSense pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MovementSample:
    """Single labeled movement sample represented as flat numeric features."""

    label: str
    features: list[float]


@dataclass(frozen=True)
class TrainingDataset:
    """Collection of labeled movement samples for model training."""

    samples: list[MovementSample]
    feature_names: list[str]


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable model bundle written by the training pipeline."""

    algorithm: str
    feature_names: list[str]
    centroids_by_label: dict[str, list[float]]


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the model training routine."""

    dataset_path: Path
    output_path: Path

