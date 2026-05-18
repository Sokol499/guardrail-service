"""Local HuggingFace transformer moderation."""

from functools import lru_cache
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Map toxic-bert labels to risk categories
LABEL_TO_CATEGORY: dict[str, str] = {
    "toxic": "toxic_language",
    "severe_toxic": "toxic_language",
    "obscene": "sexual_content",
    "threat": "violence",
    "insult": "harassment",
    "identity_hate": "hate_speech",
}


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str):
    """Lazy-load the transformers pipeline (cached singleton)."""
    from transformers import pipeline

    logger.info("loading_local_model", model=model_name)
    return pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        device=-1,  # CPU; use 0 for GPU in production
    )


class TransformerModerator:
    """Classify toxicity using a local transformer model."""

    def __init__(self, model_name: Optional[str] = None, threshold: Optional[float] = None):
        settings = get_settings()
        self.model_name = model_name or settings.local_model_name
        self.threshold = threshold if threshold is not None else settings.local_model_threshold
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = _load_pipeline(self.model_name)
        return self._pipe

    def classify(self, text: str) -> dict:
        """
        Run multi-label classification and return top scores.

        Returns:
            dict with keys: top_label, top_score, all_scores, category
        """
        # Truncate for model limits
        truncated = text[:512]
        results = self.pipe(truncated)

        # Handle both single-label and multi-label output formats
        if isinstance(results, list) and results and isinstance(results[0], list):
            scores = {r["label"]: r["score"] for r in results[0]}
        elif isinstance(results, list):
            scores = {r["label"]: r["score"] for r in results}
        else:
            scores = {results["label"]: results["score"]}

        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]
        category = LABEL_TO_CATEGORY.get(top_label, "toxic_language")

        return {
            "top_label": top_label,
            "top_score": top_score,
            "all_scores": scores,
            "category": category,
            "flagged": top_score >= self.threshold,
        }
