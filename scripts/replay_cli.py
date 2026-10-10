"""CLI replay test: stream one recording and print detections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stepsense.agents import DetectionAgent
from stepsense.movement_model import MovementModel, read_dataset
from stepsense.stream import replay_recording


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to raw dataset CSV/TSV.")
    parser.add_argument("--model", default=Path("models/model.joblib"), type=Path, help="Path to trained model.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection probability threshold.")
    parser.add_argument("--consecutive", type=int, default=2, help="Consecutive windows needed for event start.")
    parser.add_argument("--cooldown", type=float, default=30.0, help="Cooldown seconds to avoid duplicate alerts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MovementModel.load(args.model)
    df = read_dataset(args.dataset)
    participant_column = _resolve_participant_column(df)
    demo_df = df[df[participant_column].astype(str) == model.bundle.participant_holdout].copy()

    timeline: list[tuple[float, float]] = []
    for time_seconds, probability, _raw_signal_chunk, ground_truth_label in replay_recording(
        demo_df, model, speed=1.0, sleep_enabled=False
    ):
        timeline.append((time_seconds, probability))
        print(
            f"time={time_seconds:7.2f}s prob={probability:.3f} ground_truth={ground_truth_label}"
        )

    detector = DetectionAgent(consecutive_windows=args.consecutive, cooldown_seconds=args.cooldown)
    events = detector.run(timeline, threshold=args.threshold)
    print("\nDetected events:")
    for event in events:
        print(
            f"- start={event.start_time_seconds:.2f}s "
            f"duration={event.duration_seconds:.2f}s "
            f"mean_confidence={event.mean_confidence:.3f}"
        )


def _resolve_participant_column(df):
    lowered = {column.lower(): column for column in df.columns}
    for candidate in ["subject", "participant", "participant_id", "person", "id"]:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("Unable to resolve participant column.")


if __name__ == "__main__":
    main()

