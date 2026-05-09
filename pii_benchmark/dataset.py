"""Dataset loading for PII redaction benchmark suites."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    text: str
    pii: list[str]
    anchors: list[str]
    tags: list[str] = field(default_factory=list)


def load_jsonl(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw: dict[str, Any] = json.loads(line)
            try:
                sample = Sample(
                    id=str(raw["id"]),
                    text=str(raw["text"]),
                    pii=list(raw.get("pii", [])),
                    anchors=list(raw.get("anchors", [])),
                    tags=list(raw.get("tags", [])),
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no}: missing required field {exc}") from exc
            samples.append(sample)
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples


def seed_text(samples: list[Sample]) -> str:
    return "\n\n".join(sample.text for sample in samples)

