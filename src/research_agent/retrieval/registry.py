"""Retriever and reranker registries.

Two separate registries live here because ``Retriever`` and ``Reranker`` are
distinct protocols.  Factory functions in implementation modules self-register
via the appropriate decorator at module level — no manual wiring required.
"""

from __future__ import annotations

from research_agent.registry import Registry
from research_agent.retrieval.protocols import Reranker, Retriever

retriever_registry: Registry[Retriever] = Registry(
    label="retriever",
    package="research_agent.retrieval",
)

reranker_registry: Registry[Reranker] = Registry(
    label="reranker",
    package="research_agent.retrieval",
)
