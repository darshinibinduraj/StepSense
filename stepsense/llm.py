"""Small LLM wrapper with env-driven provider selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Protocol):
    """Contract for text generation providers used by StepSense agents."""

    def generate(self, prompt: str) -> str:
        """Generate text for a prompt."""


@dataclass(frozen=True)
class MockLLMProvider:
    """Deterministic canned-response provider for no-key local runs."""

    def generate(self, prompt: str) -> str:
        normalized = " ".join(prompt.split())
        return json.dumps(
            {
                "provider": "mock",
                "response": f"Mock response for: {normalized[:120]}",
            }
        )


@dataclass(frozen=True)
class UnconfiguredRemoteProvider:
    """Placeholder provider for anthropic/openai without SDK dependencies."""

    provider_name: str

    def generate(self, prompt: str) -> str:
        del prompt
        raise RuntimeError(
            f"Provider '{self.provider_name}' is selected but remote API wiring is "
            "not configured in this prototype environment. Set LLM_PROVIDER=mock "
            "for deterministic local runs."
        )


def get_provider_name() -> str:
    raw_value = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if raw_value not in {"mock", "anthropic", "openai"}:
        raise ValueError("LLM_PROVIDER must be one of: mock, anthropic, openai")
    return raw_value


def build_provider() -> LLMProvider:
    """Build provider selected by `LLM_PROVIDER` env var."""

    provider_name = get_provider_name()
    if provider_name == "mock":
        return MockLLMProvider()
    return UnconfiguredRemoteProvider(provider_name=provider_name)

