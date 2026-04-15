"""Application settings loaded from environment variables.

All configuration lives here. No ``os.getenv()`` calls are permitted anywhere
else in the codebase — import ``get_settings()`` and use the returned
``Settings`` instance instead.

Usage::

    from research_agent.config import get_settings

    settings = get_settings()
    print(settings.qdrant_url)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated, immutable application settings.

    All fields are loaded from environment variables (case-insensitive).
    Required secrets are validated at startup — the application will refuse
    to start if any are missing or contain known placeholder values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Raise on extra env vars to catch typos in variable names.
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM Inference (HuggingFace Hub / Featherless AI)
    # ------------------------------------------------------------------

    hf_token: SecretStr = Field(
        description=(
            "HuggingFace user access token used as the API key when routing "
            "inference through Featherless AI."
        ),
    )
    llm_model: str = Field(
        default="Qwen/Qwen3-32B",
        description="Model identifier served via Featherless AI through HuggingFace Hub.",
    )
    llm_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens to generate per LLM completion.",
    )

    # ------------------------------------------------------------------
    # Vector Store (Qdrant)
    # ------------------------------------------------------------------

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL of the Qdrant instance (local or cloud).",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "Qdrant API key. Required for cloud deployments; "
            "omit for unauthenticated local instances."
        ),
    )
    qdrant_collection: str = Field(
        default="research_chunks",
        description="Name of the Qdrant collection used to store research chunks.",
    )
    qdrant_vector_size: int = Field(
        default=1024,
        gt=0,
        description="Embedding vector dimension — must match the embedding model output.",
    )

    # ------------------------------------------------------------------
    # Retrieval — Embedding
    # ------------------------------------------------------------------

    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description=(
            "HuggingFace model ID used for dense embedding generation. "
            "Must produce vectors of length matching ``qdrant_vector_size``."
        ),
    )

    # ------------------------------------------------------------------
    # Memory Service (Mem0)
    # ------------------------------------------------------------------

    mem0_api_key: SecretStr = Field(
        description="Mem0 API key for the persistent memory service.",
    )
    mem0_retention_days: int = Field(
        default=30,
        gt=0,
        description="Number of days before Mem0 session data expires.",
    )

    # ------------------------------------------------------------------
    # Web Acquisition (Firecrawl via MCP)
    # ------------------------------------------------------------------

    firecrawl_api_key: SecretStr = Field(
        description="Firecrawl API key for web content acquisition via MCP.",
    )

    # ------------------------------------------------------------------
    # FastAPI / Authentication
    # ------------------------------------------------------------------

    secret_key: SecretStr = Field(
        description=(
            "Secret key used to sign JWT tokens. "
            "Must be at least 32 bytes of random data. "
            'Generate with: python -c "import secrets; print(secrets.token_hex(32))"'
        ),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm.",
    )
    jwt_expiry_minutes: int = Field(
        default=60,
        gt=0,
        description="JWT token expiry in minutes.",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    retrieval_top_k: int = Field(
        default=20,
        gt=0,
        description="Number of candidates to retrieve from Qdrant before reranking.",
    )
    retrieval_rerank_top_n: int = Field(
        default=5,
        gt=0,
        description="Number of results to return after FlashRank reranking.",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity.",
    )
    log_json: bool = Field(
        default=False,
        description="Emit structured JSON logs when True (recommended in production).",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_secrets(self) -> Settings:
        """Reject missing or placeholder secret values at startup."""
        _PLACEHOLDERS = frozenset(
            {
                "changeme",
                "your-key-here",
                "your_key_here",
                "todo",
                "xxx",
                "replace-me",
                "replace_me",
                "<your-token>",
                "<token>",
            }
        )

        _REQUIRED_SECRETS: tuple[str, ...] = (
            "hf_token",
            "mem0_api_key",
            "firecrawl_api_key",
            "secret_key",
        )

        for field_name in _REQUIRED_SECRETS:
            value: SecretStr | None = getattr(self, field_name, None)
            if value is None or value.get_secret_value().strip() == "":
                raise ValueError(
                    f"Required secret '{field_name}' is not set. "
                    "Check your .env file or Railway Variables panel."
                )
            if value.get_secret_value().lower() in _PLACEHOLDERS:
                raise ValueError(
                    f"'{field_name}' contains a placeholder value — "
                    "replace it with a real secret before starting the application."
                )

        _secret_key = self.secret_key.get_secret_value()
        if len(_secret_key.encode()) < 32:
            raise ValueError(
                "SECRET_KEY must be at least 32 bytes. "
                'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance.

    The ``Settings`` object is constructed once on first call and cached for
    the lifetime of the process. In tests, call ``get_settings.cache_clear()``
    after patching environment variables to force re-initialisation.
    """
    return Settings()
