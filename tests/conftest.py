"""Shared pytest fixtures and configuration."""

import pytest

from research_agent.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear the lru_cache on get_settings before and after every test.

    This ensures that patched environment variables are picked up by each
    test in isolation without leaking state between tests.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
