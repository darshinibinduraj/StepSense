"""Load curated research source metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sources(path: Path | None = None) -> list[dict[str, Any]]:
    source_path = path or Path("research/sources.json")
    if not source_path.exists():
        return []
    return json.loads(source_path.read_text(encoding="utf-8"))

