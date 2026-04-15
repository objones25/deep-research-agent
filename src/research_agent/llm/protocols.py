"""Core protocols and value objects for the LLM subsystem.

All external dependencies are hidden behind these protocols. Business logic
(agent nodes, API routes) must depend only on what is defined here — never
on concrete implementation classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class Message:
    """An immutable value object representing a single chat message.

    ``role`` is constrained to the three standard OpenAI-compatible roles.
    Validated at construction time so callers fail fast on bad data.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"Invalid role: {self.role!r}. Must be one of: system, user, assistant"
            )


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for a chat-style language model client.

    All I/O methods are ``async``; implementations must not block the
    event loop.
    """

    async def complete(self, messages: list[Message]) -> str:
        """Return the model's text response to *messages*.

        Parameters
        ----------
        messages:
            Ordered list of conversation turns.  Must not be empty.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        ValueError
            If *messages* is empty or the model returns no content.
        """
        ...
