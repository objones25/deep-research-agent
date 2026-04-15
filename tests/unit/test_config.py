"""Unit tests for research_agent.config.

All tests patch environment variables directly — no .env file is read.
Each test is isolated: the get_settings cache is cleared by the autouse
fixture in conftest.py.
"""

from __future__ import annotations

from typing import Any

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


def _make_settings(overrides: dict[str, str] | None = None, **kwargs: Any) -> Settings:
    """Construct a Settings instance from _VALID_ENV, applying any overrides.

    ``_env_file=None`` prevents pydantic-settings from loading the project's
    real ``.env`` file so tests remain isolated from developer secrets.
    """
    env = {**_VALID_ENV, **(overrides or {}), **kwargs}
    return Settings(_env_file=None, **{k.lower(): v for k, v in env.items()})


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
        assert settings.qdrant_vector_size == 1024

    def test_default_embedding_model(self) -> None:
        settings = _make_settings()
        assert settings.embedding_model == "BAAI/bge-large-en-v1.5"

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
        settings = Settings(_env_file=None, **{k.lower(): v for k, v in env.items()})
        assert settings.qdrant_api_key is None

    def test_default_langchain_tracing_v2_is_false(self) -> None:
        settings = _make_settings()
        assert settings.langchain_tracing_v2 is False

    def test_default_langchain_api_key_is_none(self) -> None:
        settings = _make_settings()
        assert settings.langchain_api_key is None

    def test_default_app_version(self) -> None:
        settings = _make_settings()
        assert settings.app_version == "0.1.0"

    def test_default_environment(self) -> None:
        settings = _make_settings()
        assert settings.environment == "dev"


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

    def test_custom_embedding_model(self) -> None:
        settings = _make_settings(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

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

    def test_langchain_tracing_v2_enabled(self) -> None:
        settings = _make_settings(langchain_tracing_v2="true")
        assert settings.langchain_tracing_v2 is True

    def test_langchain_api_key_set(self) -> None:
        settings = _make_settings(langchain_api_key="ls__abc123def456")
        assert settings.langchain_api_key is not None
        assert settings.langchain_api_key.get_secret_value() == "ls__abc123def456"

    def test_custom_environment_staging(self) -> None:
        settings = _make_settings(environment="staging")
        assert settings.environment == "staging"

    def test_custom_environment_prod(self) -> None:
        settings = _make_settings(environment="prod")
        assert settings.environment == "prod"

    def test_invalid_environment_raises(self) -> None:
        with pytest.raises((ValueError, Exception)):
            _make_settings(environment="production")


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

    def test_langchain_api_key_not_in_repr(self) -> None:
        settings = _make_settings(langchain_api_key="ls__supersecret")
        assert "ls__supersecret" not in repr(settings)


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
            Settings(_env_file=None, **{k.lower(): v for k, v in env.items()})

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
# Computed properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSettingsComputedProperties:
    def test_firecrawl_mcp_endpoint_embeds_api_key(self) -> None:
        settings = _make_settings()
        assert "firecrawl-key-abc123" in settings.firecrawl_mcp_endpoint

    def test_firecrawl_mcp_endpoint_ends_with_v2_mcp(self) -> None:
        settings = _make_settings()
        assert settings.firecrawl_mcp_endpoint.endswith("/v2/mcp")

    def test_firecrawl_mcp_endpoint_strips_trailing_slash_from_base(self) -> None:
        settings = _make_settings(firecrawl_mcp_url="https://mcp.firecrawl.dev/")
        endpoint = settings.firecrawl_mcp_endpoint
        assert "//" not in endpoint.replace("https://", "").replace("http://", "")

    def test_firecrawl_mcp_endpoint_full_url_structure(self) -> None:
        settings = _make_settings(firecrawl_mcp_url="https://mcp.firecrawl.dev")
        expected = "https://mcp.firecrawl.dev/firecrawl-key-abc123/v2/mcp"
        assert settings.firecrawl_mcp_endpoint == expected

    def test_qdrant_connect_kwargs_without_api_key(self) -> None:
        env = {k: v for k, v in _VALID_ENV.items() if k != "QDRANT_API_KEY"}
        settings = Settings(_env_file=None, **{k.lower(): v for k, v in env.items()})
        assert settings.qdrant_connect_kwargs == {"url": "http://localhost:6333"}

    def test_qdrant_connect_kwargs_with_api_key_includes_url(self) -> None:
        settings = _make_settings()
        assert settings.qdrant_connect_kwargs["url"] == "http://localhost:6333"

    def test_qdrant_connect_kwargs_with_api_key_includes_api_key(self) -> None:
        settings = _make_settings()
        assert settings.qdrant_connect_kwargs["api_key"] == "qdrant-key-abc123"

    def test_qdrant_connect_kwargs_custom_url(self) -> None:
        settings = _make_settings(qdrant_url="https://cloud.qdrant.io")
        assert settings.qdrant_connect_kwargs["url"] == "https://cloud.qdrant.io"

    def test_qdrant_connect_kwargs_without_api_key_has_no_api_key_entry(self) -> None:
        env = {k: v for k, v in _VALID_ENV.items() if k != "QDRANT_API_KEY"}
        settings = Settings(_env_file=None, **{k.lower(): v for k, v in env.items()})
        assert "api_key" not in settings.qdrant_connect_kwargs


# ---------------------------------------------------------------------------
# Provider selector fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSettingsProviderFields:
    def test_default_llm_provider(self) -> None:
        settings = _make_settings()
        assert settings.llm_provider == "huggingface"

    def test_custom_llm_provider(self) -> None:
        settings = _make_settings(llm_provider="openai")
        assert settings.llm_provider == "openai"

    def test_default_memory_provider(self) -> None:
        settings = _make_settings()
        assert settings.memory_provider == "mem0"

    def test_custom_memory_provider(self) -> None:
        settings = _make_settings(memory_provider="zep")
        assert settings.memory_provider == "zep"

    def test_default_retriever_type(self) -> None:
        settings = _make_settings()
        assert settings.retriever_type == "hybrid"

    def test_custom_retriever_type(self) -> None:
        settings = _make_settings(retriever_type="dense_only")
        assert settings.retriever_type == "dense_only"

    def test_default_reranker_type(self) -> None:
        settings = _make_settings()
        assert settings.reranker_type == "flashrank"

    def test_custom_reranker_type(self) -> None:
        settings = _make_settings(reranker_type="cross_encoder")
        assert settings.reranker_type == "cross_encoder"

    def test_default_enabled_tools(self) -> None:
        settings = _make_settings()
        assert settings.enabled_tools == ["firecrawl"]

    def test_custom_enabled_tools_single(self) -> None:
        settings = _make_settings(enabled_tools=["web_search"])
        assert settings.enabled_tools == ["web_search"]

    def test_custom_enabled_tools_multiple(self) -> None:
        settings = _make_settings(enabled_tools=["firecrawl", "web_search"])
        assert settings.enabled_tools == ["firecrawl", "web_search"]

    def test_enabled_tools_empty_list(self) -> None:
        settings = _make_settings(enabled_tools=[])
        assert settings.enabled_tools == []


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

    def test_cache_cleared_between_tests_by_fixture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Demonstrates that the autouse fixture in conftest clears the cache."""
        for k, v in _VALID_ENV.items():
            monkeypatch.setenv(k, v)
        # This call would fail if a stale cached instance from a previous test
        # with different env vars were returned.
        settings = get_settings()
        assert settings.hf_token.get_secret_value() == _VALID_ENV["HF_TOKEN"]
