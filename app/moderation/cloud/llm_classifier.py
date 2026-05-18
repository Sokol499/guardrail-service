"""OpenAI LLM classifier with structured JSON output."""

import json
from typing import Optional
import time

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.enums import Decision, RiskCategory, RiskLevel

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a content moderation classifier for conversational AI assistant responses.

Given an assistant response and relevant policy excerpts, classify the content.

Risk categories (use exactly one primary category):
hate_speech, extremism, violence, self_harm, medical_advice, legal_advice,
pii_leakage, sexual_content, harassment, fraud, malware, jailbreak_attempts,
prompt_injection, political_manipulation, disinformation, toxic_language,
privacy_violations, unsafe_instructions, none

Decisions:
- ALLOW: content is safe
- WARN: borderline or low-confidence violation; may proceed with caution
- BLOCK: clear policy violation; must not be shown to users

Risk levels: LOW, MEDIUM, HIGH

You MUST return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.
Ensure the JSON is complete and properly closed.

Return EXACTLY this JSON schema:
{
  "decision": "ALLOW|WARN|BLOCK",
  "category": "string",
  "risk_level": "LOW|MEDIUM|HIGH",
  "explanation": "string"
}
"""


class LLMClassifier:
    """Classify content using OpenAI with policy context."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_model
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            settings = get_settings()
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for cloud moderation. "
                    "Set it in .env or use approach=local."
                )
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=10.0,  # seconds
            )
        return self._client

    def _build_user_prompt(
        self,
        content: str,
        policy_chunks: list[dict],
        context: Optional[str] = None,
    ) -> str:
        policies_text = "\n\n".join(
            f"[Policy {c['id']}] ({c['metadata'].get('category', 'general')})\n{c['text']}"
            for c in policy_chunks
        )
        parts = [
            "## Relevant Policies",
            policies_text or "(No policies retrieved)",
            "",
            "## Assistant Response to Evaluate",
            content,
        ]
        if context:
            parts.extend(["", "## Conversation Context", context])
        return "\n".join(parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    async def classify(
        self,
        content: str,
        policy_chunks: list[dict],
        context: Optional[str] = None,
    ) -> dict:
        user_prompt = self._build_user_prompt(content, policy_chunks, context)
        start = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            top_p=0.1,
            max_tokens=500
        )

        latency_ms = (time.perf_counter() - start) * 1000
        
        logger.info(
            "llm_classification_complete",
            latency_ms=round(latency_ms, 2),
            model=self.model)

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("llm_json_parse_error", error=str(e), raw=raw[:200])
            parsed = {
                "decision": "WARN",
                "category": "none",
                "risk_level": "MEDIUM",
                "explanation": "Classification parse error; defaulting to WARN.",
                "confidence": 0.0,
            }

        return self._normalize(parsed)

    def _normalize(self, data: dict) -> dict:
        """Validate and coerce LLM output to domain enums."""
        decision_str = str(data.get("decision", "ALLOW")).upper()
        category_str = str(data.get("category", "none")).lower()
        level_str = str(data.get("risk_level", "LOW")).upper()

        try:
            decision = Decision(decision_str)
        except ValueError:
            decision = Decision.WARN

        try:
            category = RiskCategory(category_str)
        except ValueError:
            category = RiskCategory.NONE

        try:
            risk_level = RiskLevel(level_str)
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        return {
            "decision": decision,
            "category": category,
            "risk_level": risk_level,
            "explanation": str(data.get("explanation", "No explanation provided.")),
            "confidence": float(data.get("confidence", 0.5)),
        }
