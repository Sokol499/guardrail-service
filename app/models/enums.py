"""Domain enumerations for moderation."""

from enum import Enum


class Decision(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskCategory(str, Enum):
    HATE_SPEECH = "hate_speech"
    EXTREMISM = "extremism"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_ADVICE = "legal_advice"
    PII_LEAKAGE = "pii_leakage"
    SEXUAL_CONTENT = "sexual_content"
    HARASSMENT = "harassment"
    FRAUD = "fraud"
    MALWARE = "malware"
    JAILBREAK_ATTEMPTS = "jailbreak_attempts"
    PROMPT_INJECTION = "prompt_injection"
    POLITICAL_MANIPULATION = "political_manipulation"
    DISINFORMATION = "disinformation"
    TOXIC_LANGUAGE = "toxic_language"
    PRIVACY_VIOLATIONS = "privacy_violations"
    UNSAFE_INSTRUCTIONS = "unsafe_instructions"
    NONE = "none"


class ModerationApproach(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
