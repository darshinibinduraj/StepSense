"""Baseline movement model utilities for StepSense."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from stepsense.schemas import ModelArtifact, MovementSample, TrainingDataset


class MovementModel:
    """Nearest-centroid classifier trained on labeled movement features."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def train(cls, dataset: TrainingDataset) -> "MovementModel":
        if not dataset.samples:
            raise ValueError("Training dataset is empty.")

        dimensions = len(dataset.feature_names)
        if dimensions == 0:
            raise ValueError("Feature list is empty.")

        sums_by_label: dict[str, list[float]] = defaultdict(lambda: [0.0] * dimensions)
        counts_by_label: dict[str, int] = defaultdict(int)

        for sample in dataset.samples:
            if len(sample.features) != dimensions:
                raise ValueError("Inconsistent feature dimensions across samples.")

            counts_by_label[sample.label] += 1
            for index, value in enumerate(sample.features):
                sums_by_label[sample.label][index] += value

        centroids_by_label: dict[str, list[float]] = {}
        for label, summed_values in sums_by_label.items():
            count = counts_by_label[label]
            centroids_by_label[label] = [value / count for value in summed_values]

        artifact = ModelArtifact(
            algorithm="nearest_centroid",
            feature_names=dataset.feature_names,
            centroids_by_label=centroids_by_label,
        )
        return cls(artifact)

    def predict(self, features: list[float]) -> str:
        if len(features) != len(self.artifact.feature_names):
            raise ValueError("Feature dimensions do not match the trained model.")

        best_label = ""
        best_distance = math.inf

        for label, centroid in self.artifact.centroids_by_label.items():
            distance = _euclidean_distance(features, centroid)
            if distance < best_distance:
                best_distance = distance
                best_label = label

        if not best_label:
            raise RuntimeError("Model has no labels available for prediction.")

        return best_label

    def to_json(self) -> str:
        payload = {
            "algorithm": self.artifact.algorithm,
            "feature_names": self.artifact.feature_names,
            "centroids_by_label": self.artifact.centroids_by_label,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, model_path: Path) -> "MovementModel":
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        artifact = ModelArtifact(
            algorithm=payload["algorithm"],
            feature_names=list(payload["feature_names"]),
            centroids_by_label={
                str(label): [float(value) for value in centroid]
                for label, centroid in payload["centroids_by_label"].items()
            },
        )
        return cls(artifact)


def build_dataset(feature_names: list[str], rows: list[dict[str, str]]) -> TrainingDataset:
    """Convert parsed CSV rows into a typed training dataset."""

    samples: list[MovementSample] = []
    for row in rows:
        label = row["label"]
        features = [float(row[name]) for name in feature_names]
        samples.append(MovementSample(label=label, features=features))
    return TrainingDataset(samples=samples, feature_names=feature_names)


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

