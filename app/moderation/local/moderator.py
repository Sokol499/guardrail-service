"""Local OSS moderation pipeline."""

import re
from typing import Optional

from app.core.logging import get_logger
from app.models.enums import ModerationApproach, RiskCategory
from app.moderation.base import BaseModerator, ModerationResult
from app.moderation.local.pii_detector import PIIDetector
from app.moderation.local.risk_scorer import RiskScorer, RiskSignal
from app.moderation.local.semantic_detector import SemanticJailbreakDetector
from app.moderation.local.transformer import TransformerModerator

logger = get_logger(__name__)

# Keyword heuristics for categories not covered well by toxic-bert
KEYWORD_RULES: list[tuple[RiskCategory, re.Pattern[str], float]] = [
    (
        RiskCategory.SELF_HARM,
        re.compile(
            r"(?i)\b(suicide|self[- ]?harm|kill myself|end my life)\b"
        ),
        0.9,
    ),
    (
        RiskCategory.MEDICAL_ADVICE,
        re.compile(
            r"(?i)\b(take \d+ mg|prescribe|diagnos(e|is)|dosage|medication for)\b"
        ),
        0.7,
    ),
    (
        RiskCategory.LEGAL_ADVICE,
        re.compile(
            r"(?i)\b(you should sue|legal advice|statute of limitations|"
            r"file a lawsuit)\b"
        ),
        0.65,
    ),
    (
        RiskCategory.MALWARE,
        re.compile(
            r"(?i)\b(keylogger|ransomware|exploit kit|reverse shell|"
            r"privilege escalation)\b"
        ),
        0.85,
    ),
    (
        RiskCategory.FRAUD,
        re.compile(
            r"(?i)\b(credit card fraud|phishing|steal credentials|"
            r"wire transfer scam)\b"
        ),
        0.8,
    ),
    (
        RiskCategory.UNSAFE_INSTRUCTIONS,
        re.compile(
            r"(?i)\b(how to (make|build) (a )?bomb|build a pipe bomb|pipe bomb|"
            r"synthesize (meth|fentanyl))\b"
        ),
        0.95,
    ),
    (
        RiskCategory.SELF_HARM,
        re.compile(r"(?i)\b(end your life|methods to (die|kill yourself))\b"),
        0.92,
    ),
]


class LocalModerator(BaseModerator):
    approach = ModerationApproach.LOCAL

    def __init__(self) -> None:
        self.transformer = TransformerModerator()
        self.pii_detector = PIIDetector()
        self.semantic_detector = SemanticJailbreakDetector()
        self.risk_scorer = RiskScorer()

    async def moderate(
        self,
        content: str,
        context: Optional[str] = None,
    ) -> ModerationResult:
        logger.info("local_moderation_start", content_length=len(content))
        signals: list[RiskSignal] = []
        full_text = f"{context or ''}\n{content}".strip()

        # 1. Transformer toxicity
        tox = self.transformer.classify(content)
        # Include transformer signal when flagged OR score is materially elevated
        if tox["flagged"] or tox["top_score"] >= 0.55:
            signals.append(
                RiskSignal(
                    source="transformer",
                    category=RiskCategory(tox["category"]),
                    score=tox["top_score"],
                    detail=f"Label '{tox['top_label']}' score {tox['top_score']:.3f}",
                )
            )

        # 2. PII detection
        pii_matches = self.pii_detector.detect_pii(content)
        pii_category = self.pii_detector.categorize_pii_risk(pii_matches)
        if pii_category and pii_matches:
            pii_score = min(0.65 + 0.1 * len(pii_matches), 1.0)
            signals.append(
                RiskSignal(
                    source="pii_detector",
                    category=pii_category,
                    score=pii_score,
                    detail=f"Found {len(pii_matches)} PII match(es): "
                    f"{', '.join(m.pii_type for m in pii_matches[:3])}",
                )
            )

        # 3. Jailbreak / injection patterns
        jailbreak_hits = self.pii_detector.detect_jailbreak(full_text)
        if jailbreak_hits:
            label, snippet = jailbreak_hits[0]
            cat = (
                RiskCategory.PROMPT_INJECTION
                if "injection" in label or "prompt" in label
                else RiskCategory.JAILBREAK_ATTEMPTS
            )
            signals.append(
                RiskSignal(
                    source="rule_engine",
                    category=cat,
                    score=0.92,
                    detail=f"Pattern '{label}' matched: '{snippet[:80]}'",
                )
            )

        # 4. Semantic jailbreak similarity
        sem_detected, sem_score, sem_match = self.semantic_detector.detect(full_text)
        if sem_detected and sem_match:
            signals.append(
                RiskSignal(
                    source="semantic_jailbreak",
                    category=RiskCategory.JAILBREAK_ATTEMPTS,
                    score=min(sem_score, 1.0),
                    detail=(
                        f"Semantic similarity {sem_score:.3f} to canonical "
                        f"example: '{sem_match[:80]}'"
                    ),
                )
            )

        # 5. Keyword heuristics
        for category, pattern, score in KEYWORD_RULES:
            if pattern.search(content):
                signals.append(
                    RiskSignal(
                        source="keyword_rules",
                        category=category,
                        score=score,
                        detail=f"Keyword rule matched for {category.value}",
                    )
                )

        decision, category, level, confidence, explanation = self.risk_scorer.aggregate(
            signals
        )

        logger.info(
            "local_moderation_complete",
            decision=decision.value,
            category=category.value,
            risk_level=level.value,
        )

        return ModerationResult(
            decision=decision,
            category=category,
            risk_level=level,
            explanation=explanation,
            confidence=round(confidence, 4),
        )
