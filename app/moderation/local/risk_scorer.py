"""Lightweight risk scoring and decision mapping."""

from dataclasses import dataclass
from typing import Optional
from app.core.logging import get_logger

from app.models.enums import Decision, RiskCategory, RiskLevel

logger = get_logger(__name__)

@dataclass
class RiskSignal:
    source: str
    category: RiskCategory
    score: float  # 0.0 - 1.0
    detail: str


class RiskScorer:
    """Aggregate multiple signals into a final decision."""

    # Category-specific severity weights
    CATEGORY_WEIGHTS: dict[RiskCategory, float] = {
        RiskCategory.JAILBREAK_ATTEMPTS: 1.0,
        RiskCategory.PROMPT_INJECTION: 1.0,
        RiskCategory.SELF_HARM: 1.0,
        RiskCategory.MALWARE: 0.95,
        RiskCategory.PII_LEAKAGE: 0.9,
        RiskCategory.VIOLENCE: 0.9,
        RiskCategory.HATE_SPEECH: 0.85,
        RiskCategory.FRAUD: 0.85,
        RiskCategory.UNSAFE_INSTRUCTIONS: 0.85,
        RiskCategory.SEXUAL_CONTENT: 0.8,
        RiskCategory.HARASSMENT: 0.75,
        RiskCategory.TOXIC_LANGUAGE: 0.6,
        RiskCategory.PRIVACY_VIOLATIONS: 0.7,
        RiskCategory.NONE: 0.0,
    }

    def score_to_level(self, score: float) -> RiskLevel:
        if score >= 0.75:
            return RiskLevel.HIGH
        if score >= 0.45:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def level_to_decision(self, level: RiskLevel) -> Decision:
        if level == RiskLevel.HIGH:
            return Decision.BLOCK
        if level == RiskLevel.MEDIUM:
            return Decision.WARN
        return Decision.ALLOW

    def aggregate(
        self,
        signals: list[RiskSignal],
    ) -> tuple[Decision, RiskCategory, RiskLevel, float, str]:
        if not signals:
            return (
                Decision.ALLOW,
                RiskCategory.NONE,
                RiskLevel.LOW,
                0.0,
                "No risk signals detected.",
            )

        weighted_scores: list[tuple[RiskSignal, float]] = []
        for sig in signals:
            weight = self.CATEGORY_WEIGHTS.get(sig.category, 0.5)
            weighted_scores.append((sig, sig.score * weight))

        ensemble_score = min(sum(score for _, score in weighted_scores) / max(len(weighted_scores), 1), 1.0,)

        best_sig, best_weighted = max(weighted_scores, key=lambda x: x[1])

        raw_score = ensemble_score
        level = self.score_to_level(ensemble_score)
        decision = self.level_to_decision(level)

        explanations = [f"{s.source}: {s.detail} (score={s.score:.2f})" for s in signals]
        explanation = (
            f"Ensemble moderation detected multiple risk signals. "
            f"Primary risk from {best_sig.source} "
            f"[{best_sig.category.value}] "
            f"with aggregate score {ensemble_score:.2f}. "
            + "; ".join(explanations[:3])
        )

        logger.info(
            "risk_aggregation_complete",
            signals=len(signals),
            ensemble_score=round(ensemble_score, 3),
            primary_category=best_sig.category.value,
            decision=decision.value,
        )

        return decision, best_sig.category, level, raw_score, explanation
