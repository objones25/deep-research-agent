"""Dependency container for the research agent.

``AgentDependencies`` is the single wiring point for all protocol-typed
components.  Construct it once with concrete implementations and pass it to
``create_graph``.  Swapping a component (e.g. replacing the retriever) means
constructing ``AgentDependencies`` with a different implementation — no changes
to the graph or node code are required.

All fields are typed to protocol interfaces, enforcing Dependency Inversion:
high-level modules (graph, nodes) never depend on concrete implementations.

Usage::

    from research_agent.agent.dependencies import AgentDependencies

    deps = AgentDependencies(
        llm_client=my_llm,
        retriever=my_retriever,
        reranker=my_reranker,
        memory_service=my_memory,
        tools=(search_tool, scrape_tool),
    )
    graph = create_graph(deps)
"""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.llm.protocols import LLMClient
from research_agent.memory.protocols import MemoryService
from research_agent.retrieval.protocols import Reranker, Retriever
from research_agent.tools.protocols import Tool


@dataclass(frozen=True)
class AgentDependencies:
    """Immutable container of all protocol-typed agent components.

    Construct once at application startup and pass to ``create_graph``.
    ``__post_init__`` validates protocol compliance immediately, giving a clear
    ``TypeError`` at wiring time rather than a cryptic ``AttributeError``
    deep inside a graph node.

    Attributes:
        llm_client: Chat-completion client satisfying ``LLMClient``.
        retriever: Document retrieval backend satisfying ``Retriever``.
        reranker: Result reranker satisfying ``Reranker``.
        memory_service: Persistent memory backend satisfying ``MemoryService``.
        tools: Immutable sequence of tool implementations satisfying ``Tool``.
    """

    llm_client: LLMClient
    retriever: Retriever
    reranker: Reranker
    memory_service: MemoryService
    tools: tuple[Tool, ...]

    def __post_init__(self) -> None:
        _checks = (
            ("llm_client", self.llm_client, LLMClient),
            ("retriever", self.retriever, Retriever),
            ("reranker", self.reranker, Reranker),
            ("memory_service", self.memory_service, MemoryService),
        )
        for name, value, protocol in _checks:
            if not isinstance(value, protocol):
                raise TypeError(
                    f"{name!r} must implement the {protocol.__name__} protocol, "
                    f"got {type(value).__name__!r} instead."
                )

    def tools_as_list(self) -> list[Tool]:
        """Return ``tools`` as a list.

        Convenience helper for callees (e.g. ``make_tool_node``) that expect
        a ``list[Tool]`` rather than a tuple.
        """
        return list(self.tools)
