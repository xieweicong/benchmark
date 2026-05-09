"""Configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML config requires `pip install .[yaml]`.") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be an object: {path}")
    return raw


def parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    values = [part.strip() for part in value.split(",") if part.strip()]
    return values or None


def selected_models(config: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    models = list(config.get("models", []))
    if names:
        wanted = set(names)
        selected = [
            model for model in models
            if model.get("name") in wanted or model.get("type") in wanted or model.get("model_id") in wanted
        ]
        missing = wanted.difference(
            str(value)
            for model in selected
            for value in (model.get("name"), model.get("type"), model.get("model_id"))
            if value
        )
        if missing:
            raise ValueError(f"Unknown model selector(s): {', '.join(sorted(missing))}")
        return selected
    return [model for model in models if model.get("enabled", True)]

