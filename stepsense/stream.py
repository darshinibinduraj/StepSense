"""Replay movement recordings window-by-window as a live stream."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pandas as pd

from stepsense.movement_model import (
    STEP_SAMPLES,
    extract_windows,
)
from stepsense.schemas import WindowRecord


def replay_recording(
    df: pd.DataFrame,
    model,
    *,
    speed: float = 1.0,
    sleep_enabled: bool = False,
) -> Iterator[tuple[float, float, pd.DataFrame, int]]:
    """
    Replay a recording and yield:
    (time_seconds, window_probability, raw_signal_chunk, ground_truth_label).
    """

    if speed <= 0:
        raise ValueError("speed must be > 0")

    windows = extract_windows(df)
    if not windows:
        return

    feature_frame = pd.DataFrame([record.features for record in windows], columns=model.bundle.feature_names)
    probabilities = model.predict_probabilities(feature_frame)
    step_seconds = STEP_SAMPLES / model.bundle.sample_rate_hz

    for record, probability in zip(windows, probabilities):
        raw_chunk = _extract_raw_chunk(df, record)
        yield record.start_seconds, float(probability), raw_chunk, int(record.is_freezing)
        if sleep_enabled:
            time.sleep(step_seconds / speed)


def _extract_raw_chunk(df: pd.DataFrame, record: WindowRecord) -> pd.DataFrame:
    participant_column = _resolve_participant_column(df)
    participant_slice = df[df[participant_column].astype(str) == record.participant_id].reset_index(drop=True)
    return participant_slice.iloc[record.start_index : record.end_index].copy()


def _resolve_participant_column(df: pd.DataFrame) -> str:
    lowered = {column.lower(): column for column in df.columns}
    for candidate in ["subject", "participant", "participant_id", "person", "id"]:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("Unable to resolve participant column for streaming.")

