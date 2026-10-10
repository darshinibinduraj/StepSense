# StepSense

StepSense is an interactive mobile app project focused on analyzing labeled movement
recordings, gathering relevant research context, and supporting evidence-grounded
reporting about freezing-of-gait patterns.

## Disclaimer

This is a **research prototype, not a medical device**. It is not clinically validated.
The current model detects **possible freezing episodes** in replayed recordings; it does
not predict falls and does not prevent falls.

## Current implementation scope

- LOSO evaluation and training pipeline in `scripts/train.py`
- Window/feature extraction + RandomForest and threshold baseline in `stepsense/movement_model.py`
- Replay stream generator in `stepsense/stream.py`
- Multi-agent alert workflow and logging in `stepsense/agents.py` and `stepsense/orchestrator.py`
- Deterministic mock LLM provider as default (`LLM_PROVIDER=mock`)

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run a quick module compile check:

   ```bash
   python3 -m compileall stepsense scripts
   ```

## Dataset

Reference dataset:
https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

Training expects a table with participant id, labels, and numeric sensor columns.
Label handling follows the spec: windows with >50% label 0 are dropped; a window is
freezing if >=25% of labels are 2.

## Train and evaluate

```bash
python3 scripts/train.py --dataset /path/to/daphnet.csv
```

Outputs:

- `models/eval_results.json`
- `models/model.joblib`

## Replay CLI test

```bash
python3 scripts/replay_cli.py \
  --dataset /path/to/daphnet.csv \
  --model models/model.joblib
```

## Environment

Copy `.env.example` to `.env` and set:

- `LLM_PROVIDER=mock|openai|anthropic`

Default is `mock` for deterministic, no-key local runs.
