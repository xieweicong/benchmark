"""Small dependency-free token counter used for benchmark buckets.

The goal is a stable cross-machine unit, not exact parity with any model
tokenizer. Model adapters may additionally report their own tokenizer counts.
"""

from __future__ import annotations

import re


TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
    r"|[A-Za-z0-9_@./:+-]+"
    r"|[^\s]",
    re.UNICODE,
)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in TOKEN_RE.finditer(text)]


def token_count(text: str) -> int:
    return len(token_spans(text))


def truncate_to_tokens(text: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    spans = token_spans(text)
    if len(spans) <= target_tokens:
        return text
    return text[: spans[target_tokens - 1][1]]


def build_sized_text(seed: str, target_tokens: int) -> str:
    if not seed.strip():
        raise ValueError("Cannot build benchmark text from an empty seed.")

    text = seed
    while token_count(text) < target_tokens:
        text = f"{text}\n{seed}"
    return truncate_to_tokens(text, target_tokens)

