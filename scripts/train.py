"""Train StepSense model, run LOSO evaluation, and save artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stepsense.movement_model import (
    evaluate_loso,
    evaluation_to_dict,
    read_dataset,
    train_final_model,
)
from stepsense.schemas import TrainingConfig


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to Daphnet-format CSV/TSV dataset.",
    )
    parser.add_argument(
        "--model-output",
        default=Path("models/model.joblib"),
        type=Path,
        help="Output path for trained demo model.",
    )
    parser.add_argument(
        "--eval-output",
        default=Path("models/eval_results.json"),
        type=Path,
        help="Output path for LOSO evaluation JSON.",
    )
    parser.add_argument(
        "--demo-holdout-participant",
        default=None,
        help="Optional participant id to hold out for demo recording.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        dataset_path=args.dataset,
        model_output_path=args.model_output,
        eval_output_path=args.eval_output,
        demo_holdout_participant=args.demo_holdout_participant,
    )


def main() -> None:
    config = parse_args()
    df = read_dataset(config.dataset_path)
    evaluation = evaluate_loso(df)
    evaluation_dict = evaluation_to_dict(evaluation)

    config.eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.eval_output_path.write_text(json.dumps(evaluation_dict, indent=2), encoding="utf-8")

    model, holdout = train_final_model(
        df,
        demo_holdout_participant=config.demo_holdout_participant,
    )
    model.save(config.model_output_path)

    print("LOSO evaluation (RF mean):", evaluation.rf_mean)
    print("LOSO evaluation (baseline mean):", evaluation.baseline_mean)
    print("Skipped participants:", evaluation.skipped_participants_no_freezing)
    print(f"Saved evaluation to {config.eval_output_path}")
    print(f"Saved model to {config.model_output_path}")
    print(f"Demo holdout participant: {holdout}")


if __name__ == "__main__":
    main()

