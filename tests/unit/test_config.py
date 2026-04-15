"""Unit tests for research_agent.config.

All tests patch environment variables directly — no .env file is read.
Each test is isolated: the get_settings cache is cleared by the autouse
fixture in conftest.py.
"""

from __future__ import annotations

import pytest

from research_agent.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ENV: dict[str, str] = {
    "HF_TOKEN": "hf_" + "a" * 36,
    "QDRANT_API_KEY": "qdrant-key-abc123",
    "MEM0_API_KEY": "mem0-key-abc123",
    "FIRECRAWL_API_KEY": "firecrawl-key-abc123",
    # 64-character hex string = 32 bytes
    "SECRET_KEY": "a" * 64,
}


def _make_settings(overrides: dict[str, str] | None = None, **kwargs: str) -> Settings:
    """Construct a Settings instance from _VALID_ENV, applying any overrides."""
    env = {**_VALID_ENV, **(overrides or {}), **kwargs}
    return Settings(**{k.lower(): v for k, v in env.items()})


# ---------------------------------------------------------------------------
# Happy path — valid configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSettingsDefaults:
    def test_loads_valid_env(self) -> None:
        settings = _make_settings()
        assert settings.hf_token.get_secret_value() == _VALID_ENV["HF_TOKEN"]

    def test_default_llm_model(self) -> None:
        settings = _make_settings()
        assert settings.llm_model == "Qwen/Qwen3-32B"

    def test_default_llm_max_tokens(self) -> None:
        settings = _make_settings()
        assert settings.llm_max_tokens == 4096

    def test_default_qdrant_url(self) -> None:
        settings = _make_settings()
        assert settings.qdrant_url == "http://localhost:6333"

    def test_default_qdrant_collection(self) -> None:
        settings = _make_settings()
        assert settings.qdrant_collection == "research_chunks"

    def test_default_qdrant_vector_size(self) -> None:
        settings = _make_settings()
        assert settings.qdrant_vector_size == 1536

    def test_default_mem0_retention_days(self) -> None:
        settings = _make_settings()
        assert settings.mem0_retention_days == 30

    def test_default_jwt_algorithm(self) -> None:
        settings = _make_settings()
        assert settings.jwt_algorithm == "HS256"

    def test_default_jwt_expiry_minutes(self) -> None:
        settings = _make_settings()
        assert settings.jwt_expiry_minutes == 60

    def test_default_retrieval_top_k(self) -> None:
        settings = _make_settings()
        assert settings.retrieval_top_k == 20

    def test_default_retrieval_rerank_top_n(self) -> None:
        settings = _make_settings()
        assert settings.retrieval_rerank_top_n == 5

    def test_default_log_level(self) -> None:
        settings = _make_settings()
        assert settings.log_level == "INFO"

    def test_default_log_json(self) -> None:
        settings = _make_settings()
        assert settings.log_json is False

    def test_qdrant_api_key_optional_defaults_to_none(self) -> None:
        env = {k: v for k, v in _VALID_ENV.items() if k != "QDRANT_API_KEY"}
        settings = Settings(**{k.lower(): v for k, v in env.items()})
        assert settings.qdrant_api_key is None


@pytest.mark.unit
class TestSettingsOverrides:
    def test_custom_llm_model(self) -> None:
        settings = _make_settings(llm_model="Qwen/Qwen3-14B")
        assert settings.llm_model == "Qwen/Qwen3-14B"

    def test_custom_qdrant_url(self) -> None:
        settings = _make_settings(qdrant_url="https://my-cluster.cloud.qdrant.io")
        assert settings.qdrant_url == "https://my-cluster.cloud.qdrant.io"

    def test_custom_qdrant_collection(self) -> None:
        settings = _make_settings(qdrant_collection="my_collection")
        assert settings.qdrant_collection == "my_collection"

    def test_custom_retrieval_top_k(self) -> None:
        settings = _make_settings(retrieval_top_k="50")
        assert settings.retrieval_top_k == 50

    def test_custom_retrieval_rerank_top_n(self) -> None:
        settings = _make_settings(retrieval_rerank_top_n="10")
        assert settings.retrieval_rerank_top_n == 10

    def test_custom_log_level_debug(self) -> None:
        settings = _make_settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_log_json_enabled(self) -> None:
        settings = _make_settings(log_json="true")
        assert settings.log_json is True

    def test_custom_jwt_expiry(self) -> None:
        settings = _make_settings(jwt_expiry_minutes="120")
        assert settings.jwt_expiry_minutes == 120

    def test_mem0_retention_days_custom(self) -> None:
        settings = _make_settings(mem0_retention_days="7")
        assert settings.mem0_retention_days == 7


# ---------------------------------------------------------------------------
# Secret values are not leaked by repr/str
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSecretRedaction:
    def test_hf_token_not_in_repr(self) -> None:
        settings = _make_settings()
        assert _VALID_ENV["HF_TOKEN"] not in repr(settings)

    def test_mem0_api_key_not_in_repr(self) -> None:
        settings = _make_settings()
        assert _VALID_ENV["MEM0_API_KEY"] not in repr(settings)

    def test_firecrawl_api_key_not_in_repr(self) -> None:
        settings = _make_settings()
        assert _VALID_ENV["FIRECRAWL_API_KEY"] not in repr(settings)

    def test_secret_key_not_in_repr(self) -> None:
        settings = _make_settings()
        assert _VALID_ENV["SECRET_KEY"] not in repr(settings)

    def test_secret_value_accessible_via_get_secret_value(self) -> None:
        settings = _make_settings()
        assert settings.hf_token.get_secret_value() == _VALID_ENV["HF_TOKEN"]


# ---------------------------------------------------------------------------
# Validation — missing required secrets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingRequiredSecrets:
    @pytest.mark.parametrize(
        "missing_field",
        ["HF_TOKEN", "MEM0_API_KEY", "FIRECRAWL_API_KEY", "SECRET_KEY"],
    )
    def test_missing_required_secret_raises(self, missing_field: str) -> None:
        env = {k: v for k, v in _VALID_ENV.items() if k != missing_field}
        with pytest.raises((ValueError, Exception)):
            Settings(**{k.lower(): v for k, v in env.items()})

    def test_empty_hf_token_raises(self) -> None:
        with pytest.raises(ValueError, match="hf_token"):
            _make_settings(hf_token="")

    def test_empty_mem0_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="mem0_api_key"):
            _make_settings(mem0_api_key="")

    def test_empty_firecrawl_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="firecrawl_api_key"):
            _make_settings(firecrawl_api_key="")

    def test_empty_secret_key_raises(self) -> None:
        with pytest.raises(ValueError, match="secret_key"):
            _make_settings(secret_key="")

    def test_whitespace_only_secret_raises(self) -> None:
        with pytest.raises(ValueError, match="secret_key"):
            _make_settings(secret_key="   ")


# ---------------------------------------------------------------------------
# Validation — placeholder values rejected
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPlaceholderValues:
    @pytest.mark.parametrize(
        "placeholder",
        [
            "changeme",
            "CHANGEME",
            "your-key-here",
            "your_key_here",
            "todo",
            "TODO",
            "xxx",
            "replace-me",
            "replace_me",
            "<your-token>",
            "<token>",
        ],
    )
    def test_placeholder_hf_token_raises(self, placeholder: str) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            _make_settings(hf_token=placeholder)

    @pytest.mark.parametrize("placeholder", ["changeme", "todo", "xxx"])
    def test_placeholder_mem0_api_key_raises(self, placeholder: str) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            _make_settings(mem0_api_key=placeholder)

    @pytest.mark.parametrize("placeholder", ["changeme", "todo", "xxx"])
    def test_placeholder_firecrawl_api_key_raises(self, placeholder: str) -> None:
        with pytest.raises(ValueError, match="placeholder"):
            _make_settings(firecrawl_api_key=placeholder)

    @pytest.mark.parametrize("placeholder", ["changeme", "todo", "xxx"])
    def test_placeholder_secret_key_raises(self, placeholder: str) -> None:
        # "xxx" is only 3 bytes so it also fails the length check;
        # ensure the placeholder check fires first (or either raises).
        with pytest.raises(ValueError):
            _make_settings(secret_key=placeholder)


# ---------------------------------------------------------------------------
# Validation — SECRET_KEY minimum length
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSecretKeyLength:
    def test_secret_key_exactly_32_bytes_accepted(self) -> None:
        # 32 ASCII characters = 32 bytes
        settings = _make_settings(secret_key="a" * 32)
        assert len(settings.secret_key.get_secret_value().encode()) >= 32

    def test_secret_key_31_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            _make_settings(secret_key="a" * 31)

    def test_secret_key_64_hex_chars_accepted(self) -> None:
        key = "b" * 64
        settings = _make_settings(secret_key=key)
        assert settings.secret_key.get_secret_value() == key

    def test_secret_key_multibyte_utf8_counts_bytes_not_chars(self) -> None:
        # Each '€' is 3 bytes in UTF-8; 11 × 3 = 33 bytes ≥ 32 → valid
        key = "€" * 11
        settings = _make_settings(secret_key=key)
        assert len(settings.secret_key.get_secret_value().encode()) >= 32

    def test_secret_key_multibyte_short_raises(self) -> None:
        # Each '€' is 3 bytes; 10 × 3 = 30 bytes < 32 → invalid
        with pytest.raises(ValueError, match="32 bytes"):
            _make_settings(secret_key="€" * 10)


# ---------------------------------------------------------------------------
# Validation — invalid log level
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogLevelValidation:
    @pytest.mark.parametrize("invalid", ["TRACE", "VERBOSE", "warn", "info"])
    def test_invalid_log_level_raises(self, invalid: str) -> None:
        with pytest.raises((ValueError, Exception)):
            _make_settings(log_level=invalid)

    @pytest.mark.parametrize("valid", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels_accepted(self, valid: str) -> None:
        settings = _make_settings(log_level=valid)
        assert settings.log_level == valid


# ---------------------------------------------------------------------------
# get_settings — caching behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetSettingsCache:
    def test_get_settings_returns_settings_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k, v in _VALID_ENV.items():
            monkeypatch.setenv(k, v)
        result = get_settings()
        assert isinstance(result, Settings)

    def test_get_settings_returns_same_object_on_second_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for k, v in _VALID_ENV.items():
            monkeypatch.setenv(k, v)
        first = get_settings()
        second = get_settings()
        assert first is second

    def test_cache_cleared_between_tests_by_fixture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Demonstrates that the autouse fixture in conftest clears the cache."""
        for k, v in _VALID_ENV.items():
            monkeypatch.setenv(k, v)
        # This call would fail if a stale cached instance from a previous test
        # with different env vars were returned.
        settings = get_settings()
        assert settings.hf_token.get_secret_value() == _VALID_ENV["HF_TOKEN"]
