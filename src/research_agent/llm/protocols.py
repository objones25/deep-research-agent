"""Core protocols for the LLM subsystem.

All external dependencies are hidden behind these protocols. Business logic
(agent nodes, API routes) must depend only on what is defined here — never
on concrete implementation classes.

Value objects (Message) live in ``research_agent.models.research`` because
they flow up through the layer stack (llm → agent).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from research_agent.models.research import Message

__all__ = ["LLMClient", "Message"]


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
