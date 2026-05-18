"""Application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "guardrail-service"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    local_model_name: str = "unitary/toxic-bert"
    local_model_threshold: float = 0.65

    chroma_persist_dir: str = "./data/chroma"
    policy_data_path: str = "./data/policies/policy_chunks.json"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_jailbreak_threshold: float = 0.62

    default_approach: Literal["local", "cloud"] = "local"
    rag_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
