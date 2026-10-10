"""Multi-agent workflow components for StepSense alerts."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from stepsense.llm import build_provider, get_provider_name
from stepsense.schemas import DetectionEvent, ReviewResult, TriageDecision

BANNED_CLAIMS = [
    "fall prevented",
    "fall predicted",
    "diagnosis",
    "clinically validated",
    "cueing works for everyone",
    "prevents falls",
]


class DetectionAgent:
    """Deterministic detection agent from probability windows."""

    def __init__(self, consecutive_windows: int = 2, cooldown_seconds: float = 30.0) -> None:
        self.consecutive_windows = consecutive_windows
        self.cooldown_seconds = cooldown_seconds

    def run(
        self, timeline: list[tuple[float, float]], *, threshold: float
    ) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        streak_start_index: int | None = None
        last_event_start = float("-inf")

        for index, (time_seconds, probability) in enumerate(timeline):
            if probability >= threshold:
                if streak_start_index is None:
                    streak_start_index = index
                streak_length = index - streak_start_index + 1
                if streak_length >= self.consecutive_windows:
                    start_time = timeline[streak_start_index][0]
                    if start_time - last_event_start >= self.cooldown_seconds:
                        window_probabilities = [item[1] for item in timeline[streak_start_index : index + 1]]
                        duration = max(1, streak_length) * 1.0
                        events.append(
                            DetectionEvent(
                                start_time_seconds=start_time,
                                duration_seconds=duration,
                                mean_confidence=float(sum(window_probabilities) / len(window_probabilities)),
                            )
                        )
                        last_event_start = start_time
                    streak_start_index = None
            else:
                streak_start_index = None
        return events


class TriageAgent:
    """LLM-backed triage with deterministic fallback."""

    def __init__(self) -> None:
        self.provider = build_provider()

    def run(
        self, event: DetectionEvent, recent_events_count: int, false_alarm_cost: str = "medium"
    ) -> TriageDecision:
        fallback = _fallback_triage(event=event, recent_events_count=recent_events_count, false_alarm_cost=false_alarm_cost)
        prompt = (
            "Return JSON {send_alert, urgency, reasoning} for event triage. "
            f"event={asdict(event)} recent_events_count={recent_events_count} false_alarm_cost={false_alarm_cost}"
        )
        try:
            raw_output = self.provider.generate(prompt)
            payload = json.loads(raw_output)
            if {"send_alert", "urgency", "reasoning"}.issubset(payload):
                urgency = str(payload["urgency"]).lower()
                if urgency in {"low", "medium", "high"}:
                    return TriageDecision(
                        send_alert=bool(payload["send_alert"]),
                        urgency=urgency,
                        reasoning=str(payload["reasoning"]),
                    )
        except Exception:
            pass
        return fallback


class MessageAgent:
    """LLM-backed message drafting with mock-mode overclaim demo behavior."""

    def __init__(self) -> None:
        self.provider = build_provider()
        self.provider_name = get_provider_name()

    def run(self, event: DetectionEvent, *, revision_round: int = 0) -> dict[str, str]:
        if self.provider_name == "mock" and revision_round == 0:
            return {
                "caregiver_message": (
                    "Possible freezing episode detected at "
                    f"{_format_hhmm(event.start_time_seconds)}, lasting about "
                    f"{event.duration_seconds:.1f} seconds. This prevents falls."
                ),
                "cue_note": "Play rhythmic cue now.",
            }

        prompt = (
            "Write JSON with caregiver_message and cue_note using hedged language only. "
            f"event={asdict(event)}"
        )
        try:
            payload = json.loads(self.provider.generate(prompt))
            if {"caregiver_message", "cue_note"}.issubset(payload):
                return {
                    "caregiver_message": str(payload["caregiver_message"]),
                    "cue_note": str(payload["cue_note"]),
                }
        except Exception:
            pass

        return {
            "caregiver_message": (
                "Possible freezing episode detected at "
                f"{_format_hhmm(event.start_time_seconds)}, lasting about {event.duration_seconds:.1f} seconds."
            ),
            "cue_note": "Consider a short rhythmic cue.",
        }


class ReviewAgent:
    """Checks drafts against allowed-claims policy and sources context."""

    def run(self, draft_message: str, sources: list[dict[str, Any]]) -> ReviewResult:
        lowered = draft_message.lower()
        issues = [claim for claim in BANNED_CLAIMS if claim in lowered]
        if "possible freezing episode" not in lowered:
            issues.append("message must include the phrase 'possible freezing episode'")

        if issues:
            revised = draft_message.replace("This prevents falls.", "").strip()
            if "possible freezing episode" not in revised.lower():
                revised = f"Possible freezing episode detected. {revised}".strip()
            if not revised.endswith("."):
                revised += "."
            return ReviewResult(approved=False, issues=issues, revised_message=revised)

        source_count = len(sources)
        revised_ok = draft_message
        if source_count == 0:
            revised_ok += " (No curated sources loaded.)"
        return ReviewResult(approved=True, issues=[], revised_message=revised_ok)


def _format_hhmm(time_seconds: float) -> str:
    reference = datetime.utcfromtimestamp(max(0.0, time_seconds))
    return reference.strftime("%H:%M")


def _fallback_triage(event: DetectionEvent, recent_events_count: int, false_alarm_cost: str) -> TriageDecision:
    confidence = event.mean_confidence
    if confidence >= 0.8:
        urgency = "high"
    elif confidence >= 0.6:
        urgency = "medium"
    else:
        urgency = "low"

    if false_alarm_cost == "high" and urgency == "low":
        send_alert = False
    else:
        send_alert = confidence >= 0.5 and recent_events_count < 5

    return TriageDecision(
        send_alert=send_alert,
        urgency=urgency,
        reasoning=(
            "Rule-based fallback used because LLM output was unavailable or invalid. "
            f"confidence={confidence:.2f}, recent_events={recent_events_count}, false_alarm_cost={false_alarm_cost}"
        ),
    )

