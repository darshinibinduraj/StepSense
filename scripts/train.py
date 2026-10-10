"""Train and persist a baseline StepSense movement model from CSV data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stepsense.movement_model import MovementModel, build_dataset
from stepsense.schemas import TrainingConfig


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to CSV dataset with a 'label' column and numeric feature columns.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for serialized model JSON artifact.",
    )
    args = parser.parse_args()
    return TrainingConfig(dataset_path=args.dataset, output_path=args.output)


def read_rows(dataset_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Dataset CSV has no header row.")
        if "label" not in reader.fieldnames:
            raise ValueError("Dataset CSV must include a 'label' column.")

        feature_names = [name for name in reader.fieldnames if name != "label"]
        if not feature_names:
            raise ValueError("Dataset CSV must include at least one feature column.")

        rows = list(reader)

    if not rows:
        raise ValueError("Dataset CSV has no sample rows.")

    return feature_names, rows


def main() -> None:
    config = parse_args()
    feature_names, rows = read_rows(config.dataset_path)
    dataset = build_dataset(feature_names=feature_names, rows=rows)
    model = MovementModel.train(dataset)
    model.save(config.output_path)
    print(f"Saved model to {config.output_path}")


if __name__ == "__main__":
    main()

