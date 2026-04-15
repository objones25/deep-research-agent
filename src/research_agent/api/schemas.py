"""Pydantic request/response schemas for the FastAPI layer.

These are distinct from the internal dataclasses in ``models/research.py``.
Internal models are frozen dataclasses; these are Pydantic BaseModels suitable
for FastAPI serialisation and OpenAPI schema generation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ResearchQueryRequest(BaseModel):
    """Body for ``POST /research``."""

    query: str = Field(description="The research question to investigate.")
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for cross-request memory continuity.",
    )
    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum ReAct loop iterations (1–20).",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class CitationResponse(BaseModel):
    """A single source citation returned with a research report."""

    url: str = Field(description="Source URL.")
    content_snippet: str = Field(description="Relevant excerpt from the source.")
    relevance_score: float = Field(description="Reranker relevance score (0–1).")


class ResearchReportResponse(BaseModel):
    """Body for the ``POST /research`` response."""

    query: str = Field(description="The original research question.")
    summary: str = Field(description="Synthesised research summary.")
    citations: list[CitationResponse] = Field(
        description="Source citations supporting the summary."
    )
    session_id: str = Field(description="Session identifier used for this request.")


class HealthCheckResponse(BaseModel):
    """Body for ``GET /health``."""

    status: Literal["ok", "degraded", "error"] = Field(description="Overall service health status.")
    checks: dict[str, str] = Field(
        description="Per-dependency health check results (e.g. ``{'qdrant': 'ok'}``)."
    )
    timestamp: str = Field(description="ISO 8601 UTC timestamp of the health check.")
