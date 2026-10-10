"""Behavior tests for StepSense agent safeguards."""

from __future__ import annotations

import unittest

from stepsense.agents import DetectionAgent, ReviewAgent


class ReviewAgentTests(unittest.TestCase):
    def test_review_flags_prevents_falls_claim(self) -> None:
        review = ReviewAgent().run(
            "Possible freezing episode detected at 10:01, lasting about 3 seconds. This prevents falls.",
            sources=[],
        )
        self.assertFalse(review.approved)
        self.assertTrue(any("prevents falls" in issue for issue in review.issues))


class DetectionAgentTests(unittest.TestCase):
    def test_detection_cooldown_prevents_duplicate_alert(self) -> None:
        timeline = [
            (0.0, 0.1),
            (1.0, 0.8),
            (2.0, 0.85),  # Event 1 starts at 1s
            (3.0, 0.2),
            (10.0, 0.9),
            (11.0, 0.92),  # Would be event 2 but still within 30s cooldown
            (40.0, 0.95),
            (41.0, 0.96),  # Event 2 allowed after cooldown
        ]
        detector = DetectionAgent(consecutive_windows=2, cooldown_seconds=30.0)
        events = detector.run(timeline, threshold=0.75)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].start_time_seconds, 1.0)
        self.assertEqual(events[1].start_time_seconds, 40.0)


if __name__ == "__main__":
    unittest.main()

