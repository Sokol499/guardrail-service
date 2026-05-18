"""Cloud LLM + RAG moderation pipeline."""

from typing import Optional

from app.moderation.local.moderator import LocalModerator

from app.core.logging import get_logger
from app.models.enums import ModerationApproach
from app.moderation.base import BaseModerator, ModerationResult
from app.moderation.cloud.llm_classifier import LLMClassifier
from app.moderation.cloud.rag import PolicyRAG

logger = get_logger(__name__)


class CloudModerator(BaseModerator):
    approach = ModerationApproach.CLOUD

    def __init__(self) -> None:
        self.rag = PolicyRAG()
        self.classifier = LLMClassifier()
        self.fallback = LocalModerator()

    async def moderate(
        self,
        content: str,
        context: Optional[str] = None,
    ) -> ModerationResult:
        logger.info("cloud_moderation_start", content_length=len(content))

        query = f"{context or ''}\n{content}".strip()
        policy_chunks = self.rag.retrieve(query)
        policy_ids = [c["id"] for c in policy_chunks]

        logger.info(
            "policies_retrieved",
            count=len(policy_chunks),
            ids=policy_ids,
        )

        try:
            result = await self.classifier.classify(
                content=content,
                policy_chunks=policy_chunks,
                context=context,
            )

        except Exception as e:
            logger.error(
                "cloud_moderation_failed",
                error=str(e),
            )

            logger.warning(
                "fallback_to_local_moderation",
                reason="cloud_provider_failure",
            )

            return await self.fallback.moderate(
                content=content,
                context=context,
            )

        logger.info(
            "cloud_moderation_complete",
            decision=result["decision"].value,
            category=result["category"].value,
        )

        return ModerationResult(
            decision=result["decision"],
            category=result["category"],
            risk_level=result["risk_level"],
            explanation=result["explanation"],
            confidence=result["confidence"],
            retrieved_policies=policy_ids,
        )
