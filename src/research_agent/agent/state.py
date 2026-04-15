"""LangGraph agent state definition.

``AgentState`` is the single source of truth passed through every node in the
ReAct graph.  LangGraph merges partial ``dict`` updates from each node back into
the state.  The ``Annotated[list[...], operator.add]`` fields accumulate across
updates (append-only); all other fields are last-write-wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from research_agent.models.research import (
    Message,
    ResearchReport,
    SearchResult,
    ToolResult,
)


class AgentState(TypedDict):
    """Shared mutable state threaded through every LangGraph node.

    Fields
    ------
    query:
        The raw research question supplied by the caller.
    session_id:
        Opaque identifier shared with the memory service for cross-turn recall.
    max_iterations:
        Hard cap on ReAct loop cycles — guards against runaway agents.
    iteration_count:
        Number of reason→act cycles completed so far.
    search_results:
        Most-recent retrieval results; replaced on each retrieval step.
    memories:
        Relevant memories retrieved from the memory service at loop start.
    messages:
        Full conversation history sent to the LLM.  Uses ``operator.add``
        so each node can append without knowing the current length.
    tool_results:
        Accumulated tool execution outcomes.  ``operator.add`` appends.
    tool_calls_pending:
        Structured tool-call dicts parsed from the latest LLM response, waiting
        to be dispatched by the tool node.
    final_report:
        Populated by the synthesis node when the agent is done.  ``None`` while
        the loop is still running.
    """

    query: str
    session_id: str
    max_iterations: int
    iteration_count: int
    search_results: list[SearchResult]
    memories: list[str]
    messages: Annotated[list[Message], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    tool_calls_pending: list[dict[str, Any]]
    final_report: ResearchReport | None
