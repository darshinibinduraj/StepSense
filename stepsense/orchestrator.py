"""Agent orchestrator for StepSense alert workflow."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from stepsense.agents import DetectionAgent, MessageAgent, ReviewAgent, TriageAgent
from stepsense.schemas import ActivityLogEntry, FinalAlert
from stepsense.sources import load_sources


class AlertOrchestrator:
    """Run Detection -> Triage -> Message -> Review with activity logging."""

    def __init__(self, consecutive_windows: int = 2, cooldown_seconds: float = 30.0) -> None:
        self.detection_agent = DetectionAgent(
            consecutive_windows=consecutive_windows,
            cooldown_seconds=cooldown_seconds,
        )
        self.triage_agent = TriageAgent()
        self.message_agent = MessageAgent()
        self.review_agent = ReviewAgent()

    def run(
        self,
        timeline: list[tuple[float, float]],
        *,
        threshold: float,
        false_alarm_cost: str = "medium",
    ) -> list[FinalAlert]:
        sources = load_sources()
        activity_log: list[ActivityLogEntry] = []
        events = self.detection_agent.run(timeline, threshold=threshold)

        activity_log.append(
            self._log(
                "DetectionAgent",
                {"threshold": threshold, "timeline_size": len(timeline)},
                {"event_count": len(events), "events": [asdict(event) for event in events]},
            )
        )

        outputs: list[FinalAlert] = []
        for index, event in enumerate(events):
            triage = self.triage_agent.run(event, recent_events_count=index, false_alarm_cost=false_alarm_cost)
            activity_log.append(
                self._log(
                    "TriageAgent",
                    {
                        "event": asdict(event),
                        "recent_events_count": index,
                        "false_alarm_cost": false_alarm_cost,
                    },
                    asdict(triage),
                )
            )

            draft = self.message_agent.run(event, revision_round=0)
            activity_log.append(self._log("MessageAgent", {"event": asdict(event), "revision_round": 0}, draft))

            review = self.review_agent.run(draft["caregiver_message"], sources=sources)
            activity_log.append(
                self._log(
                    "ReviewAgent",
                    {"draft_message": draft["caregiver_message"]},
                    asdict(review),
                )
            )

            final_message = review.revised_message
            if not review.approved:
                revised = self.message_agent.run(event, revision_round=1)
                # Preserve the policy-corrected phrasing if the LLM revision regresses.
                revised_message = revised["caregiver_message"]
                if "possible freezing episode" not in revised_message.lower():
                    revised_message = review.revised_message
                second_review = self.review_agent.run(revised_message, sources=sources)
                activity_log.append(
                    self._log(
                        "MessageAgent",
                        {"event": asdict(event), "revision_round": 1, "review_issues": review.issues},
                        revised,
                    )
                )
                activity_log.append(
                    self._log(
                        "ReviewAgent",
                        {"draft_message": revised_message, "revision_round": 1},
                        asdict(second_review),
                    )
                )
                review = second_review
                final_message = review.revised_message

            outputs.append(
                FinalAlert(
                    event=event,
                    triage=triage,
                    message=final_message,
                    review=review,
                    sources_cited=[str(source.get("id", "")) for source in sources],
                    activity_log=list(activity_log),
                )
            )
        return outputs

    def _log(self, agent_name: str, input_payload: dict[str, Any], output_payload: dict[str, Any]) -> ActivityLogEntry:
        return ActivityLogEntry(
            agent_name=agent_name,
            input_payload=input_payload,
            output_payload=output_payload,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

