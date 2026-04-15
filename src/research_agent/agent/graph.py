"""LangGraph agent graph factory.

``create_graph`` is the single public entry point.  It validates all injected
dependencies at construction time, wires the ReAct nodes and routing edges,
and returns a compiled ``CompiledGraph`` ready for ``ainvoke`` / ``astream``.

Graph topology
--------------

    START
      │
      ▼
   memory  ──────────────────────────────┐
      │                                  │
      ▼                                  │
   retrieve ◄─────────────────────────── │ ─── (loop back: should_continue → retrieve)
      │                                  │
      ▼                                  │
   reason                                │
      │                                  │
      ▼ should_use_tools                 │
    ┌────────────────┐                   │
    │ "tools"        │ "retrieve"        │
    ▼                ▼                   │
  tools         should_continue ─────────┘
    │               │
    │ (loop)        │ "synthesize"
    └──► reason     ▼
                synthesize
                    │
                   END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from research_agent.agent.nodes import (
    NodeFn,
    make_memory_node,
    make_reason_node,
    make_retrieval_node,
    make_synthesis_node,
    make_tool_node,
    should_continue,
    should_use_tools,
)
from research_agent.agent.state import AgentState
from research_agent.llm.protocols import LLMClient
from research_agent.memory.protocols import MemoryService
from research_agent.retrieval.protocols import Reranker, Retriever
from research_agent.tools.protocols import Tool

# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------


def _require(value: object, protocol: type, name: str) -> None:
    """Raise ``TypeError`` if ``value`` does not satisfy ``protocol``."""
    if not isinstance(value, protocol):
        raise TypeError(
            f"{name!r} must implement the {protocol.__name__} protocol, "
            f"got {type(value).__name__!r} instead."
        )


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def create_graph(
    *,
    retriever: Retriever,
    reranker: Reranker,
    memory_service: MemoryService,
    llm_client: LLMClient,
    tools: list[Tool],
) -> object:
    """Construct and compile the ReAct research agent graph.

    Parameters
    ----------
    retriever:
        Hybrid retrieval implementation satisfying ``Retriever``.
    reranker:
        Reranking implementation satisfying ``Reranker``.
    memory_service:
        Memory backend satisfying ``MemoryService``.
    llm_client:
        LLM client satisfying ``LLMClient``.
    tools:
        List of available tools, each satisfying ``Tool``.

    Returns
    -------
    CompiledGraph
        A LangGraph ``CompiledGraph`` ready for ``ainvoke`` / ``astream``.

    Raises
    ------
    TypeError
        If any dependency does not satisfy its expected protocol.
    """
    _require(retriever, Retriever, "retriever")
    _require(reranker, Reranker, "reranker")
    _require(memory_service, MemoryService, "memory_service")
    _require(llm_client, LLMClient, "llm_client")

    # ------------------------------------------------------------------
    # Build node functions by closing over injected dependencies
    # ------------------------------------------------------------------
    memory_node: NodeFn = make_memory_node(memory_service)
    retrieval_node: NodeFn = make_retrieval_node(retriever, reranker)
    reason_node: NodeFn = make_reason_node(llm_client)
    tool_node: NodeFn = make_tool_node(tools)
    synthesis_node: NodeFn = make_synthesis_node(llm_client, memory_service)

    # ------------------------------------------------------------------
    # Assemble the StateGraph
    # ------------------------------------------------------------------
    # Drop explicit generic annotation: StateGraph has 4 type params whose
    # internal names vary by LangGraph version; inference is cleaner here.
    builder = StateGraph(AgentState)

    # LangGraph's add_node overloads expect `Runnable | RunnableLike`; bare
    # async callables satisfy the runtime contract but not the type stubs.
    builder.add_node("memory", memory_node)  # type: ignore[call-overload]
    builder.add_node("retrieve", retrieval_node)  # type: ignore[call-overload]
    builder.add_node("reason", reason_node)  # type: ignore[call-overload]
    builder.add_node("tools", tool_node)  # type: ignore[call-overload]
    builder.add_node("synthesize", synthesis_node)  # type: ignore[call-overload]

    # Entry point
    builder.add_edge(START, "memory")
    builder.add_edge("memory", "retrieve")
    builder.add_edge("retrieve", "reason")

    # After reasoning: dispatch tools or decide whether to loop/synthesize
    builder.add_conditional_edges(
        "reason",
        should_use_tools,
        {"tools": "tools", "retrieve": "check_continue"},
    )

    # Shim node to allow conditional branching after should_use_tools → "retrieve"
    # LangGraph requires conditional edge targets to be node names; we use an
    # identity pass-through to route to should_continue logic.
    async def _noop(state: AgentState) -> dict[str, object]:  # noqa: RUF029
        return {}

    builder.add_node("check_continue", _noop)
    builder.add_conditional_edges(
        "check_continue",
        should_continue,
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )

    # After tool execution, loop back to reasoning
    builder.add_edge("tools", "reason")

    # Terminal node
    builder.add_edge("synthesize", END)

    return builder.compile()
