"""API request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field

from app.models.enums import Decision, ModerationApproach, RiskCategory, RiskLevel


class ModerationRequest(BaseModel):
    """Input payload: assistant response to evaluate."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=32000,
        description="Assistant response text to moderate",
        examples=["Here is how you can reset your password safely."],
    )
    approach: Optional[ModerationApproach] = Field(
        default=None,
        description="Moderation pipeline: local (OSS) or cloud (LLM+RAG)",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional conversation context for richer classification",
    )


class ModerationResponse(BaseModel):
    """Structured moderation output."""

    decision: Decision
    category: RiskCategory
    risk_level: RiskLevel
    explanation: str
    approach: ModerationApproach
    confidence: Optional[float] = Field(
        default=None,
        description="Model confidence score when available (0-1)",
    )
    retrieved_policies: Optional[list[str]] = Field(
        default=None,
        description="Policy chunk IDs used (cloud approach only)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "decision": "ALLOW",
                    "category": "none",
                    "risk_level": "LOW",
                    "explanation": "Content appears safe with no policy violations detected.",
                    "approach": "local",
                    "confidence": 0.12,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    version: str
    approaches: list[str]
