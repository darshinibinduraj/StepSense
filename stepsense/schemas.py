"""Shared dataclasses used across StepSense modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for movement model training and evaluation."""

    dataset_path: Path
    model_output_path: Path
    eval_output_path: Path
    demo_holdout_participant: str | None = None


@dataclass(frozen=True)
class WindowRecord:
    """A single window extracted from raw movement signals."""

    participant_id: str
    start_index: int
    end_index: int
    start_seconds: float
    is_freezing: int
    ankle_freeze_index: float
    features: dict[str, float]


@dataclass(frozen=True)
class SubjectMetrics:
    """Evaluation metrics for a single held-out participant."""

    participant_id: str
    balanced_accuracy: float
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True)
class LOSOEvaluationResults:
    """LOSO evaluation summary for both RF and threshold baselines."""

    rf_per_subject: list[SubjectMetrics]
    baseline_per_subject: list[SubjectMetrics]
    rf_mean: dict[str, float]
    baseline_mean: dict[str, float]
    skipped_participants_no_freezing: list[str]


@dataclass(frozen=True)
class ModelBundle:
    """Persisted model artifact used by the app and stream replay."""

    feature_names: list[str]
    participant_holdout: str
    window_size: int
    step_size: int
    sample_rate_hz: float
    freeze_probability_threshold: float


@dataclass(frozen=True)
class DetectionEvent:
    """Candidate freezing event created by the Detection Agent."""

    start_time_seconds: float
    duration_seconds: float
    mean_confidence: float


@dataclass(frozen=True)
class TriageDecision:
    """Alert triage output."""

    send_alert: bool
    urgency: str
    reasoning: str


@dataclass(frozen=True)
class ReviewResult:
    """Review output for final communication safeguards."""

    approved: bool
    issues: list[str]
    revised_message: str


@dataclass(frozen=True)
class ActivityLogEntry:
    """One logged step in the agent workflow."""

    agent_name: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass(frozen=True)
class FinalAlert:
    """Final orchestrated alert payload."""

    event: DetectionEvent
    triage: TriageDecision
    message: str
    review: ReviewResult
    sources_cited: list[str]
    activity_log: list[ActivityLogEntry]

