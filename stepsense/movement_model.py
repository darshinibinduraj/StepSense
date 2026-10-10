"""Movement model, feature extraction, and LOSO evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stepsense.schemas import LOSOEvaluationResults, ModelBundle, SubjectMetrics, WindowRecord

SAMPLE_RATE_HZ = 64.0
WINDOW_SAMPLES = 128
STEP_SAMPLES = 64


class MovementModel:
    """Random forest movement model used for window-level freezing probability."""

    def __init__(self, rf_model: RandomForestClassifier, bundle: ModelBundle) -> None:
        self.rf_model = rf_model
        self.bundle = bundle

    def predict_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return freezing probability for each extracted window."""

        windows = extract_windows(df)
        feature_frame = pd.DataFrame([record.features for record in windows], columns=self.bundle.feature_names)
        probabilities = self.predict_probabilities(feature_frame)
        return pd.DataFrame(
            {
                "start_seconds": [record.start_seconds for record in windows],
                "window_probability": probabilities,
                "ground_truth_label": [record.is_freezing for record in windows],
            }
        )

    def predict_probabilities(self, feature_frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.rf_model.predict_proba(feature_frame)
        classes = list(self.rf_model.classes_)
        if 1 in classes:
            positive_index = classes.index(1)
            return probabilities[:, positive_index]
        return np.zeros(len(feature_frame), dtype=float)

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"rf_model": self.rf_model, "bundle": asdict(self.bundle)}, model_path)

    @classmethod
    def load(cls, model_path: Path) -> "MovementModel":
        payload = joblib.load(model_path)
        bundle = ModelBundle(**payload["bundle"])
        return cls(rf_model=payload["rf_model"], bundle=bundle)


def read_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read Daphnet-like dataset from CSV/TSV into a DataFrame."""

    frame = pd.read_csv(dataset_path, sep=None, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def evaluate_loso(df: pd.DataFrame) -> LOSOEvaluationResults:
    """Run leave-one-subject-out evaluation for RF and threshold baselines."""

    participant_column = _resolve_participant_column(df)
    rf_subject_metrics: list[SubjectMetrics] = []
    baseline_subject_metrics: list[SubjectMetrics] = []
    skipped_participants: list[str] = []

    participant_ids = sorted(df[participant_column].astype(str).unique().tolist())
    for participant_id in participant_ids:
        test_df = df[df[participant_column].astype(str) == participant_id]
        train_df = df[df[participant_column].astype(str) != participant_id]
        test_windows = extract_windows(test_df)
        if not test_windows:
            skipped_participants.append(participant_id)
            continue

        y_test = np.array([record.is_freezing for record in test_windows], dtype=int)
        if int(y_test.sum()) == 0:
            skipped_participants.append(participant_id)
            continue

        train_windows = extract_windows(train_df)
        if not train_windows:
            continue

        train_features = _window_features_frame(train_windows)
        y_train = np.array([record.is_freezing for record in train_windows], dtype=int)
        test_features = _window_features_frame(test_windows)

        rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        rf_model.fit(train_features, y_train)
        rf_predictions = rf_model.predict(test_features)

        ankle_train = np.array([record.ankle_freeze_index for record in train_windows], dtype=float)
        ankle_test = np.array([record.ankle_freeze_index for record in test_windows], dtype=float)
        threshold = pick_baseline_threshold(ankle_train, y_train)
        baseline_predictions = (ankle_test >= threshold).astype(int)

        rf_subject_metrics.append(_build_metrics(participant_id, y_test, rf_predictions))
        baseline_subject_metrics.append(_build_metrics(participant_id, y_test, baseline_predictions))

    return LOSOEvaluationResults(
        rf_per_subject=rf_subject_metrics,
        baseline_per_subject=baseline_subject_metrics,
        rf_mean=_mean_metrics(rf_subject_metrics),
        baseline_mean=_mean_metrics(baseline_subject_metrics),
        skipped_participants_no_freezing=skipped_participants,
    )


def train_final_model(
    df: pd.DataFrame, demo_holdout_participant: str | None = None
) -> tuple[MovementModel, str]:
    """Train final model excluding one demo participant with freezing episodes."""

    participant_column = _resolve_participant_column(df)
    holdout = demo_holdout_participant or pick_demo_holdout_participant(df)
    train_df = df[df[participant_column].astype(str) != holdout]
    train_windows = extract_windows(train_df)
    if not train_windows:
        raise ValueError("No training windows available after excluding demo holdout participant.")

    train_features = _window_features_frame(train_windows)
    y_train = np.array([record.is_freezing for record in train_windows], dtype=int)
    ankle_train = np.array([record.ankle_freeze_index for record in train_windows], dtype=float)
    threshold = pick_baseline_threshold(ankle_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf_model.fit(train_features, y_train)

    bundle = ModelBundle(
        feature_names=train_features.columns.tolist(),
        participant_holdout=holdout,
        window_size=WINDOW_SAMPLES,
        step_size=STEP_SAMPLES,
        sample_rate_hz=SAMPLE_RATE_HZ,
        freeze_probability_threshold=float(threshold),
    )
    return MovementModel(rf_model=rf_model, bundle=bundle), holdout


def pick_demo_holdout_participant(df: pd.DataFrame) -> str:
    """Pick an unseen participant that has at least one freezing window."""

    participant_column = _resolve_participant_column(df)
    participants = sorted(df[participant_column].astype(str).unique().tolist())
    for participant_id in participants:
        participant_windows = extract_windows(df[df[participant_column].astype(str) == participant_id])
        if any(window.is_freezing == 1 for window in participant_windows):
            return participant_id
    raise ValueError("No participant with freezing windows was found for demo holdout.")


def extract_windows(df: pd.DataFrame) -> list[WindowRecord]:
    """Create 2-second windows with 1-second step and spec label logic."""

    if df.empty:
        return []

    participant_column = _resolve_participant_column(df)
    label_column = _resolve_label_column(df)
    signal_columns = _resolve_signal_columns(df, participant_column=participant_column, label_column=label_column)
    time_column = _resolve_time_column(df)

    records: list[WindowRecord] = []
    for participant_id, participant_df in df.groupby(participant_column):
        participant_df = participant_df.reset_index(drop=True)
        for start_index in range(0, len(participant_df) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            end_index = start_index + WINDOW_SAMPLES
            window_df = participant_df.iloc[start_index:end_index]

            labels = window_df[label_column].astype(int).to_numpy()
            if np.mean(labels == 0) > 0.5:
                continue

            is_freezing = 1 if np.mean(labels == 2) >= 0.25 else 0
            feature_values = _window_features(window_df, signal_columns)
            start_seconds = _window_start_seconds(window_df, start_index, time_column)
            records.append(
                WindowRecord(
                    participant_id=str(participant_id),
                    start_index=start_index,
                    end_index=end_index,
                    start_seconds=start_seconds,
                    is_freezing=is_freezing,
                    ankle_freeze_index=float(feature_values["ankle_freeze_index"]),
                    features=feature_values,
                )
            )
    return records


def pick_baseline_threshold(ankle_freeze_index: np.ndarray, labels: np.ndarray) -> float:
    """Choose threshold on training data only with best balanced accuracy."""

    unique_values = np.unique(ankle_freeze_index)
    if unique_values.size == 0:
        return 0.0
    if unique_values.size == 1:
        return float(unique_values[0])

    candidates = np.concatenate(
        [
            unique_values,
            (unique_values[1:] + unique_values[:-1]) / 2.0,
        ]
    )
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        predictions = (ankle_freeze_index >= threshold).astype(int)
        score = _binary_balanced_accuracy(labels.astype(int), predictions.astype(int))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _window_features(window_df: pd.DataFrame, signal_columns: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for column in signal_columns:
        signal = window_df[column].astype(float).to_numpy()
        features[f"{column}_mean"] = float(np.mean(signal))
        features[f"{column}_std"] = float(np.std(signal))
        features[f"{column}_min"] = float(np.min(signal))
        features[f"{column}_max"] = float(np.max(signal))
        features[f"{column}_freeze_index"] = _freeze_index(signal, SAMPLE_RATE_HZ)

    ankle_columns = [column for column in signal_columns if "ankle" in column.lower()]
    if not ankle_columns:
        raise ValueError("No ankle signal columns found; required for threshold baseline.")
    ankle_freeze_values = [
        features[f"{column}_freeze_index"] for column in ankle_columns if f"{column}_freeze_index" in features
    ]
    features["ankle_freeze_index"] = float(np.mean(ankle_freeze_values))

    combined = window_df[signal_columns].astype(float).to_numpy()
    magnitude = np.sqrt(np.sum(np.square(combined), axis=1))
    features["accel_magnitude_mean"] = float(np.mean(magnitude))
    features["accel_magnitude_std"] = float(np.std(magnitude))
    features["accel_magnitude_min"] = float(np.min(magnitude))
    features["accel_magnitude_max"] = float(np.max(magnitude))
    features["accel_magnitude_freeze_index"] = _freeze_index(magnitude, SAMPLE_RATE_HZ)
    return features


def _freeze_index(signal: np.ndarray, sample_rate_hz: float) -> float:
    centered = signal - np.mean(signal)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    power = np.abs(spectrum) ** 2

    locomotor_band = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_band = (freqs >= 3.0) & (freqs <= 8.0)
    locomotor_power = float(np.sum(power[locomotor_band]))
    freeze_power = float(np.sum(power[freeze_band]))
    if locomotor_power <= 1e-12:
        return 0.0
    return freeze_power / locomotor_power


def _resolve_participant_column(df: pd.DataFrame) -> str:
    candidates = ["subject", "participant", "participant_id", "person", "id"]
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Could not resolve participant column from: {list(df.columns)}")


def _resolve_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "annotation", "class", "target"]
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Could not resolve label column from: {list(df.columns)}")


def _resolve_time_column(df: pd.DataFrame) -> str | None:
    lowered = {column.lower(): column for column in df.columns}
    for candidate in ["time", "timestamp", "t"]:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _resolve_signal_columns(df: pd.DataFrame, participant_column: str, label_column: str) -> list[str]:
    excluded = {
        participant_column,
        label_column,
    }
    signal_columns = []
    for column in df.columns:
        lower = column.lower()
        if column in excluded or lower in {"time", "timestamp", "t", "task"}:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            signal_columns.append(column)

    if not signal_columns:
        raise ValueError("No numeric signal columns found for feature extraction.")
    return signal_columns


def _window_features_frame(records: list[WindowRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record.features for record in records])
    return frame.reindex(sorted(frame.columns), axis=1)


def _window_start_seconds(window_df: pd.DataFrame, start_index: int, time_column: str | None) -> float:
    if time_column is None:
        return start_index / SAMPLE_RATE_HZ
    first_value = float(window_df.iloc[0][time_column])
    return first_value if np.isfinite(first_value) else start_index / SAMPLE_RATE_HZ


def _build_metrics(participant_id: str, y_true: np.ndarray, y_pred: np.ndarray) -> SubjectMetrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return SubjectMetrics(
        participant_id=participant_id,
        balanced_accuracy=float(balanced_accuracy),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
    )


def _mean_metrics(metrics: list[SubjectMetrics]) -> dict[str, float]:
    if not metrics:
        return {"balanced_accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "balanced_accuracy": float(np.mean([item.balanced_accuracy for item in metrics])),
        "f1": float(np.mean([item.f1 for item in metrics])),
        "precision": float(np.mean([item.precision for item in metrics])),
        "recall": float(np.mean([item.recall for item in metrics])),
    }


def _binary_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return (recall + specificity) / 2.0


def evaluation_to_dict(evaluation: LOSOEvaluationResults) -> dict[str, Any]:
    """Serialize LOSO evaluation dataclass to a JSON-ready dictionary."""

    return {
        "rf_per_subject": [asdict(item) for item in evaluation.rf_per_subject],
        "baseline_per_subject": [asdict(item) for item in evaluation.baseline_per_subject],
        "rf_mean": evaluation.rf_mean,
        "baseline_mean": evaluation.baseline_mean,
        "skipped_participants_no_freezing": evaluation.skipped_participants_no_freezing,
    }

