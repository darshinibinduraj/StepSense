"""LLM provider abstraction with a mock-first default implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMProvider(Protocol):
    """Contract for StepSense text-generation providers."""

    def generate(self, prompt: str) -> str:
        """Generate a response string from a prompt."""


@dataclass(frozen=True)
class MockLLMProvider:
    """Deterministic mock provider used as the default implementation."""

    prefix: str = "MOCK_STEPSENSE_RESPONSE"

    def generate(self, prompt: str) -> str:
        compact_prompt = " ".join(prompt.strip().split())
        return f"{self.prefix}: {compact_prompt}"


def build_default_provider() -> LLMProvider:
    """Return the default LLM provider (mock by spec-first policy)."""

    return MockLLMProvider()

