"""Shared value objects (result types) used across research_agent subsystems.

These are cross-cutting types that flow *up* through the layer stack:
  tools → agent → API  (ToolResult)
  retrieval → agent → API  (SearchResult)
  llm → agent  (Message)

Input DTOs and exceptions that form part of a subsystem's calling contract
(e.g. SearchInput, ScrapeInput, ToolExecutionError) live in the subsystem's
own protocols.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A single document/page returned by a retrieval or search operation."""

    content: str
    url: str
    score: float
    metadata: dict[str, str]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class Message:
    """An immutable value object representing a single chat message.

    ``role`` is constrained to the four standard OpenAI-compatible roles.
    ``"tool"`` carries tool-call results back to the model in multi-turn
    tool-use conversations. Validated at construction time so callers fail
    fast on bad data.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"Invalid role: {self.role!r}. Must be one of: system, user, assistant"
            )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """Immutable value object representing the outcome of a tool execution.

    Either ``content`` or ``error`` is populated, never both.
    ``is_error`` always reflects which field carries data.
    """

    is_error: bool
    content: str | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
