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

from research_agent.agent.dependencies import AgentDependencies
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
from research_agent.models.research import ResearchQuery, ResearchReport

# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def create_graph(deps: AgentDependencies) -> object:
    """Construct and compile the ReAct research agent graph.

    Parameters
    ----------
    deps:
        All protocol-typed agent components, pre-validated at construction
        time by :class:`~research_agent.agent.dependencies.AgentDependencies`.

    Returns
    -------
    CompiledGraph
        A LangGraph ``CompiledGraph`` ready for ``ainvoke`` / ``astream``.
    """
    # ------------------------------------------------------------------
    # Build node functions by closing over injected dependencies
    # ------------------------------------------------------------------
    memory_node: NodeFn = make_memory_node(deps.memory_service)
    retrieval_node: NodeFn = make_retrieval_node(deps.retriever, deps.reranker)
    reason_node: NodeFn = make_reason_node(deps.llm_client)
    tool_node: NodeFn = make_tool_node(deps.tools_as_list())
    synthesis_node: NodeFn = make_synthesis_node(deps.llm_client, deps.memory_service)

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


# ---------------------------------------------------------------------------
# AgentRunner implementation
# ---------------------------------------------------------------------------


class CompiledGraphRunner:
    """Wraps a LangGraph compiled graph and implements the ``AgentRunner`` protocol.

    This is the production binding between the FastAPI layer and the LangGraph
    graph. Tests inject a mock ``AgentRunner`` instead of this class to avoid
    real I/O during testing.

    Usage::

        deps = AgentDependencies(retriever=..., reranker=..., memory_service=...,
                                 llm_client=..., tools=(...,))
        runner = CompiledGraphRunner(create_graph(deps))
        report = await runner.run(query)
    """

    def __init__(self, graph: object) -> None:
        """
        Args:
            graph: A compiled LangGraph graph produced by :func:`create_graph`.
        """
        self._graph = graph

    async def run(self, query: ResearchQuery) -> ResearchReport:
        """Execute the research graph for *query* and return the final report.

        Args:
            query: The research request containing the user's question and session ID.

        Returns:
            A :class:`~research_agent.models.research.ResearchReport` with a
            summary and citations.

        Raises:
            RuntimeError: If the graph completes without producing a
                ``final_report`` (should not happen under normal operation).
        """
        result: dict[str, object] = await self._graph.ainvoke(  # type: ignore[attr-defined]
            {
                "query": query.query,
                "session_id": query.session_id,
                "max_iterations": query.max_iterations,
                "iteration_count": 0,
                "search_results": [],
                "memories": [],
                "messages": [],
                "tool_results": [],
                "tool_calls_pending": [],
                "final_report": None,
            }
        )
        report: ResearchReport | None = result.get("final_report")  # type: ignore[assignment]
        if report is None:
            raise RuntimeError(
                "Agent graph completed without producing a final report. "
                "This is a bug — check the synthesis node."
            )
        return report
