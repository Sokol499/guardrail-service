"""Pytest fixtures."""

import pytest

from app.services.guardrail_service import GuardrailService


@pytest.fixture
def guardrail_service() -> GuardrailService:
    return GuardrailService()
