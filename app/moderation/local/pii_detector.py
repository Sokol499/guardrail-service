"""Regex and rule-based PII detection."""

import re
from dataclasses import dataclass
from typing import Optional

from app.models.enums import RiskCategory


@dataclass
class PIIMatch:
    pii_type: str
    matched_text: str
    start: int
    end: int


# Common PII patterns
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        re.IGNORECASE,
    ),
    "phone_us": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "api_key": re.compile(
        r"\b(?:sk|pk)[-_](?:live|test)?[-_][A-Za-z0-9]{20,}\b",
        re.IGNORECASE,
    ),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

# Rule-based jailbreak / injection indicators
JAILBREAK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+"
            r"(instructions?|rules?|prompts?)"
        ),
    ),
    (
        "dan_mode",
        re.compile(r"(?i)\b(DAN|do anything now|jailbreak)\b"),
    ),
    (
        "role_override",
        re.compile(
            r"(?i)you are now (?:a |an )?(?:unrestricted|uncensored|evil|malicious)"
        ),
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"(?i)(reveal|show|print|output|dump|leak).{0,40}(system\s+)?prompt"
        ),
    ),
    (
        "policy_bypass",
        re.compile(
            r"(?i)(bypass|disable|override|circumvent).{0,30}(safety|content\s+policy|guardrails?|filters?)"
        ),
    ),
    (
        "decode_execute",
        re.compile(r"(?i)decode\s+and\s+execute"),
    ),
    (
        "safety_override",
        re.compile(r"(?i)(override\s+safety|without\s+restrictions|no\s+ethical\s+guidelines)"),
    ),
    (
        "activation_exploit",
        re.compile(
            r"(?i)((grandmother|grandma).{0,60}(activation|product)\s+keys?"
            r"|(activation|product)\s+keys?.{0,60}(sleep|grandmother|grandma))"
        ),
    ),
    (
        "opposite_safety",
        re.compile(
            r"(?i)opposite\s+of\s+(your\s+)?(safety|content)\s+(guidelines|policies|rules)"
        ),
    ),
    (
        "prompt_injection",
        re.compile(r"(?i)\[\s*(?:system|assistant|user)\s*\]"),
    ),
]


class PIIDetector:
    """Detect personally identifiable information and injection patterns."""

    def detect_pii(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for pii_type, pattern in PII_PATTERNS.items():
            for m in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        matched_text=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return matches

    def detect_jailbreak(self, text: str) -> list[tuple[str, str]]:
        hits: list[tuple[str, str]] = []
        for label, pattern in JAILBREAK_PATTERNS:
            m = pattern.search(text)
            if m:
                hits.append((label, m.group()))
        return hits

    def categorize_pii_risk(
        self, matches: list[PIIMatch]
    ) -> Optional[RiskCategory]:
        if not matches:
            return None
        sensitive = {"ssn", "credit_card", "api_key", "aws_key"}
        if any(m.pii_type in sensitive for m in matches):
            return RiskCategory.PII_LEAKAGE
        return RiskCategory.PRIVACY_VIOLATIONS
