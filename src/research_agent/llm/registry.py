"""LLM provider registry.

Factory functions in implementation modules (e.g. ``huggingface.py``) self-register
via ``@llm_registry.register("key")`` at module level.  The registry imports those
modules automatically on the first ``build()`` call — no manual wiring required.
"""

from __future__ import annotations

from research_agent.llm.protocols import LLMClient
from research_agent.registry import Registry

llm_registry: Registry[LLMClient] = Registry(
    label="llm",
    package="research_agent.llm",
)
