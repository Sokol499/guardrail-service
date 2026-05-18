"""Abstract moderation interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.models.enums import Decision, ModerationApproach, RiskCategory, RiskLevel


@dataclass
class ModerationResult:
    decision: Decision
    category: RiskCategory
    risk_level: RiskLevel
    explanation: str
    confidence: Optional[float] = None
    retrieved_policies: Optional[list[str]] = None


class BaseModerator(ABC):
    approach: ModerationApproach

    @abstractmethod
    async def moderate(
        self,
        content: str,
        context: Optional[str] = None,
    ) -> ModerationResult:
        """Evaluate content and return a moderation decision."""
