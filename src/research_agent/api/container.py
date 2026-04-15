"""Composition root: wire all production dependencies via registries.

``build_agent_runner`` is the sole public entry point.  It reads the
provider/type keys from ``Settings`` and delegates construction to each
subsystem's ``Registry``.  Adding a new LLM backend, retriever, or tool
requires only a ``@registry.register("key")`` factory — this file never
needs to change.

Usage::

    from research_agent.api.container import build_agent_runner
    from research_agent.config import get_settings

    runner = await build_agent_runner(get_settings())
"""

from __future__ import annotations

from research_agent.api.dependencies import AgentRunner
from research_agent.config import Settings
from research_agent.llm.registry import llm_registry
from research_agent.memory.registry import memory_registry
from research_agent.retrieval.registry import reranker_registry, retriever_registry
from research_agent.tools.protocols import Tool
from research_agent.tools.registry import tools_registry


async def build_agent_runner(settings: Settings) -> AgentRunner:
    """Construct all production dependencies and return a wired ``AgentRunner``.

    Each subsystem is built by calling ``registry.build(key, settings)`` where
    *key* comes from the relevant ``Settings`` field.  Tools are gathered from
    every key in ``settings.enabled_tools`` and flattened into a single tuple.

    Args:
        settings: Validated application settings.

    Returns:
        A :class:`~research_agent.agent.graph.CompiledGraphRunner` ready to
        handle research queries.
    """
    from research_agent.agent.dependencies import AgentDependencies
    from research_agent.agent.graph import CompiledGraphRunner, create_graph

    llm_client = await llm_registry.build(settings.llm_provider, settings)
    memory_service = await memory_registry.build(settings.memory_provider, settings)
    retriever = await retriever_registry.build(settings.retriever_type, settings)
    reranker = await reranker_registry.build(settings.reranker_type, settings)

    tools: list[Tool] = []
    for key in settings.enabled_tools:
        tool_tuple = await tools_registry.build(key, settings)
        tools.extend(tool_tuple)

    deps = AgentDependencies(
        llm_client=llm_client,
        retriever=retriever,
        reranker=reranker,
        memory_service=memory_service,
        tools=tuple(tools),
    )
    graph = create_graph(deps)
    return CompiledGraphRunner(graph)
