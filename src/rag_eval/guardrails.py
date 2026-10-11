import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    triggered: bool
    reasons: list[str]


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|override)\b.{0,40}\b(previous|prior|system|developer)\b",
            re.I,
        ),
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(reveal|show|print|repeat|leak)\b.{0,40}\b(system prompt|developer message|hidden instructions)\b",
            re.I,
        ),
    ),
    ("jailbreak_request", re.compile(r"\b(jailbreak|developer mode|dan mode)\b", re.I)),
]


def inspect_query(query: str) -> GuardrailResult:
    reasons = [name for name, pattern in _PATTERNS if pattern.search(query)]
    return GuardrailResult(triggered=bool(reasons), reasons=reasons)
