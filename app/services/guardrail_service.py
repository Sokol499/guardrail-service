"""Orchestrates moderation pipelines."""

from typing import Optional

from app.api.schemas import ModerationResponse
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import ModerationApproach
from app.moderation.base import BaseModerator, ModerationResult
from app.moderation.cloud import CloudModerator
from app.moderation.local import LocalModerator

logger = get_logger(__name__)


class GuardrailService:
    """Facade for local and cloud moderation approaches."""

    def __init__(self) -> None:
        self._moderators: dict[ModerationApproach, BaseModerator] = {}

    def _get_moderator(self, approach: ModerationApproach) -> BaseModerator:
        if approach not in self._moderators:
            if approach == ModerationApproach.LOCAL:
                self._moderators[approach] = LocalModerator()
            else:
                self._moderators[approach] = CloudModerator()
        return self._moderators[approach]

    async def moderate(
        self,
        content: str,
        approach: Optional[ModerationApproach] = None,
        context: Optional[str] = None,
    ) -> ModerationResponse:
        settings = get_settings()
        selected = approach or ModerationApproach(settings.default_approach)

        logger.info(
            "moderation_request",
            approach=selected.value,
            content_length=len(content),
        )

        moderator = self._get_moderator(selected)
        result: ModerationResult = await moderator.moderate(content, context)

        return self._to_response(result, selected)

    @staticmethod
    def _to_response(
        result: ModerationResult,
        approach: ModerationApproach,
    ) -> ModerationResponse:
        return ModerationResponse(
            decision=result.decision,
            category=result.category,
            risk_level=result.risk_level,
            explanation=result.explanation,
            approach=approach,
            confidence=result.confidence,
            retrieved_policies=result.retrieved_policies,
        )
