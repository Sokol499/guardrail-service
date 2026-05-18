"""REST API route handlers."""

from fastapi import APIRouter, Depends, HTTPException

from app import __version__
from app.api.schemas import HealthResponse, ModerationRequest, ModerationResponse
from app.core.logging import get_logger
from app.models.enums import ModerationApproach
from app.services.guardrail_service import GuardrailService

logger = get_logger(__name__)

router = APIRouter()

_service: GuardrailService | None = None


def get_guardrail_service() -> GuardrailService:
    global _service
    if _service is None:
        _service = GuardrailService()
    return _service


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Service health and available moderation approaches."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        approaches=[a.value for a in ModerationApproach],
    )


@router.post(
    "/moderate",
    response_model=ModerationResponse,
    tags=["Moderation"],
    summary="Moderate assistant response",
    description=(
        "Evaluate an assistant response and return a structured moderation decision. "
        "Use `approach=local` for OSS transformer + rules, or `approach=cloud` for "
        "OpenAI + RAG policy retrieval."
    ),
)
async def moderate(
    request: ModerationRequest,
    service: GuardrailService = Depends(get_guardrail_service),
) -> ModerationResponse:
    try:
        return await service.moderate(
            content=request.content,
            approach=request.approach,
            context=request.context,
        )
    except ValueError as e:
        logger.warning("moderation_validation_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("moderation_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Moderation pipeline failed. Check logs for details.",
        ) from e


@router.post(
    "/moderate/local",
    response_model=ModerationResponse,
    tags=["Moderation"],
    summary="Moderate using local OSS pipeline",
)
async def moderate_local(
    request: ModerationRequest,
    service: GuardrailService = Depends(get_guardrail_service),
) -> ModerationResponse:
    request.approach = ModerationApproach.LOCAL
    return await moderate(request, service)


@router.post(
    "/moderate/cloud",
    response_model=ModerationResponse,
    tags=["Moderation"],
    summary="Moderate using cloud LLM + RAG pipeline",
)
async def moderate_cloud(
    request: ModerationRequest,
    service: GuardrailService = Depends(get_guardrail_service),
) -> ModerationResponse:
    request.approach = ModerationApproach.CLOUD
    return await moderate(request, service)
