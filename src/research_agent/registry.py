"""Generic provider registry with decorator-based registration and lazy auto-discovery.

Each subsystem (llm, memory, retrieval, tools) owns a ``Registry[T]`` instance.
Factory functions self-register via ``@registry.register("key")`` at module level
in their respective implementation files.  The registry imports those modules
automatically on the first ``build()`` call — no manual wiring required.

Usage::

    # In src/research_agent/llm/registry.py
    from research_agent.registry import Registry
    from research_agent.llm.protocols import LLMClient

    llm_registry: Registry[LLMClient] = Registry(
        label="llm",
        package="research_agent.llm",
    )

    # In src/research_agent/llm/huggingface.py (bottom of file)
    from research_agent.llm.registry import llm_registry

    @llm_registry.register("huggingface")
    async def _factory(settings: Settings) -> LLMClient:
        ...

    # In container.py
    client = await llm_registry.build(settings.llm_provider, settings)
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

# Type alias for a factory: async (Settings) -> T
_Factory = Callable[..., Coroutine[Any, Any, Any]]


class Registry[T]:
    """A keyed registry of async factory functions for a single protocol type.

    Parameters
    ----------
    label:
        Human-readable name for this registry (e.g. ``"llm"``).  Used in
        error messages when an unknown key is requested.
    package:
        Fully-qualified package name whose modules are scanned for
        ``@register``-decorated factories (e.g. ``"research_agent.llm"``).
        Scanning is deferred until the first ``build()`` call.
    """

    def __init__(self, label: str, package: str) -> None:
        self._label = label
        self._package = package
        self._factories: dict[str, _Factory] = {}
        self._discovered: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, key: str) -> Callable[[_Factory], _Factory]:
        """Return a decorator that registers *factory* under *key*.

        The decorator is transparent — it returns the original factory
        unchanged so it can still be called directly in tests.

        Raises
        ------
        ValueError
            If *key* is already registered in this registry.
        """

        def _decorator(factory: _Factory) -> _Factory:
            if key in self._factories:
                raise ValueError(
                    f"[{self._label}] key '{key}' is already registered. "
                    "Each provider key must be unique within a registry."
                )
            self._factories[key] = factory
            return factory

        return _decorator

    async def build(self, key: str, settings: Any) -> T:
        """Construct and return the provider registered under *key*.

        Triggers lazy auto-discovery on the first call, then looks up the
        factory and awaits it.

        Parameters
        ----------
        key:
            The provider key (e.g. ``"huggingface"``, ``"mem0"``).
        settings:
            Application settings forwarded verbatim to the factory.

        Returns
        -------
        T
            The constructed provider instance.

        Raises
        ------
        ValueError
            If *key* is not registered after discovery.
        """
        if not self._discovered:
            self._discover()

        factory = self._factories.get(key)
        if factory is None:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"[{self._label}] unknown provider key '{key}'. " f"Available: {available}"
            )

        result: T = await factory(settings)
        return result

    # ------------------------------------------------------------------
    # Discovery (internal)
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Import every leaf module in ``self._package``.

        Importing a module fires any ``@registry.register(...)`` decorators
        at its module level, populating ``_factories`` without any manual
        wiring.  Sub-packages (``ispkg=True``) are skipped — only direct
        child modules are scanned.

        Sets ``_discovered = True`` when complete so the scan runs at most
        once per registry instance.
        """
        pkg = importlib.import_module(self._package)
        for _finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            if not ispkg:
                importlib.import_module(f"{self._package}.{name}")
        self._discovered = True
