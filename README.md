# StepSense

StepSense is an interactive mobile app project focused on analyzing labeled movement
recordings, gathering relevant research context, and supporting evidence-grounded
reporting about freezing-of-gait patterns.

## Initial repository scaffold

This repository currently contains an initial Python scaffold for:

- Data schemas (`stepsense/schemas.py`)
- LLM provider abstraction with a mock default (`stepsense/llm.py`)
- A baseline movement model implementation (`stepsense/movement_model.py`)
- A training script entrypoint (`scripts/train.py`)

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run a quick module compile check:

   ```bash
   python -m compileall stepsense scripts
   ```

## Baseline training script usage

The training script expects a CSV file containing:

- `label` column (class label)
- one or more numeric feature columns

Example:

```bash
python scripts/train.py \
  --dataset /path/to/movement_dataset.csv \
  --output /tmp/stepsense_model.json
```
