"""Shared pytest fixtures and configuration."""

import pytest

from research_agent.config import get_settings

# All environment variable names that pydantic-settings reads for Settings.
# Kept in sync with the field names in config.py.
_SETTINGS_ENV_VARS: tuple[str, ...] = (
    "HF_TOKEN",
    "LLM_MODEL",
    "LLM_MAX_TOKENS",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_VECTOR_SIZE",
    "MEM0_API_KEY",
    "MEM0_RETENTION_DAYS",
    "FIRECRAWL_API_KEY",
    "SECRET_KEY",
    "JWT_ALGORITHM",
    "JWT_EXPIRY_MINUTES",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_RERANK_TOP_N",
    "LOG_LEVEL",
    "LOG_JSON",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
)


@pytest.fixture(autouse=True)
def isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all known Settings env vars before every test.

    pydantic-settings merges init_kwargs with OS environment variables.
    Tests that construct ``Settings`` via keyword arguments (without
    explicitly passing every field) would otherwise pick up real secrets
    from the developer's shell — causing "missing secret" tests to pass
    silently and "optional default" tests to get wrong values.

    Clearing these vars ensures that only what each test explicitly
    provides is visible to the ``Settings`` constructor.
    """
    for var in _SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear the lru_cache on get_settings before and after every test.

    This ensures that patched environment variables are picked up by each
    test in isolation without leaking state between tests.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
