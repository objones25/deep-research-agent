"""Unit tests for the generic Registry class (TDD — RED phase).

Covers:
- register() decorator stores factory under key and returns it unchanged
- Duplicate key registration raises ValueError
- build() calls the registered factory with settings and returns the result
- build() with an unknown key raises ValueError that mentions the label
  and lists the available keys
- _discover() is NOT called at construction time
- _discover() IS called lazily on the first build() call
- _discover() runs exactly once, no matter how many times build() is called
- _discover() imports every non-package module found in the configured package
- _discover() skips sub-packages (ispkg=True)
- _discover() sets _discovered to True after running
"""

from __future__ import annotations

import importlib as _importlib
import pkgutil as _pkgutil
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_agent.registry import Registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings(**kwargs: Any) -> MagicMock:
    """Return a lightweight settings-shaped mock."""
    mock = MagicMock()
    for k, v in kwargs.items():
        setattr(mock, k, v)
    return mock


# ---------------------------------------------------------------------------
# register() decorator
# ---------------------------------------------------------------------------


class TestRegisterDecorator:
    def test_register_stores_factory_under_key(self) -> None:
        registry: Registry[str] = Registry(label="test", package=__name__)

        @registry.register("my_key")
        async def _factory(settings: Any) -> str:
            return "result"

        assert "my_key" in registry._factories

    def test_register_returns_factory_unchanged(self) -> None:
        """The decorator must be transparent — it must return the original callable."""
        registry: Registry[str] = Registry(label="test", package=__name__)

        async def _factory(settings: Any) -> str:
            return "hello"

        returned = registry.register("key")(_factory)
        assert returned is _factory

    def test_register_multiple_distinct_keys(self) -> None:
        registry: Registry[str] = Registry(label="test", package=__name__)

        @registry.register("a")
        async def _fa(settings: Any) -> str:
            return "a"

        @registry.register("b")
        async def _fb(settings: Any) -> str:
            return "b"

        assert "a" in registry._factories
        assert "b" in registry._factories

    def test_register_duplicate_key_raises(self) -> None:
        registry: Registry[str] = Registry(label="test", package=__name__)

        @registry.register("dup")
        async def _f1(settings: Any) -> str:
            return "first"

        with pytest.raises(ValueError, match="dup"):

            @registry.register("dup")
            async def _f2(settings: Any) -> str:
                return "second"


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


class TestBuild:
    async def test_build_calls_factory_with_settings(self) -> None:
        registry: Registry[str] = Registry(label="test", package=__name__)
        settings = _fake_settings()

        received: list[Any] = []

        @registry.register("greet")
        async def _factory(s: Any) -> str:
            received.append(s)
            return "ok"

        registry._discovered = True  # skip fs scan in unit test
        await registry.build("greet", settings)

        assert received == [settings]

    async def test_build_returns_factory_result(self) -> None:
        registry: Registry[int] = Registry(label="nums", package=__name__)

        @registry.register("forty_two")
        async def _factory(s: Any) -> int:
            return 42

        registry._discovered = True
        result = await registry.build("forty_two", _fake_settings())
        assert result == 42

    async def test_build_unknown_key_raises_value_error(self) -> None:
        registry: Registry[str] = Registry(label="test", package=__name__)

        @registry.register("known")
        async def _factory(s: Any) -> str:
            return "ok"

        registry._discovered = True

        with pytest.raises(ValueError):
            await registry.build("unknown_key", _fake_settings())

    async def test_build_error_mentions_label(self) -> None:
        """ValueError for an unknown key must include the registry label."""
        registry: Registry[str] = Registry(label="llm", package=__name__)
        registry._discovered = True

        with pytest.raises(ValueError, match="llm"):
            await registry.build("nonexistent", _fake_settings())

    async def test_build_error_lists_available_keys(self) -> None:
        """ValueError for an unknown key must list the registered keys."""
        registry: Registry[str] = Registry(label="mem", package=__name__)

        @registry.register("mem0")
        async def _f(s: Any) -> str:
            return "x"

        registry._discovered = True

        with pytest.raises(ValueError, match="mem0"):
            await registry.build("zep", _fake_settings())


# ---------------------------------------------------------------------------
# Lazy discovery
# ---------------------------------------------------------------------------


class TestLazyDiscovery:
    def test_discover_not_called_at_construction(self) -> None:
        with patch.object(Registry, "_discover") as mock_discover:
            Registry(label="x", package="some.package")
            mock_discover.assert_not_called()

    async def test_discover_called_on_first_build(self) -> None:
        """First build() must trigger _discover() exactly once."""
        call_count = 0

        def _noop_discover(self: Any) -> None:
            nonlocal call_count
            call_count += 1
            self._discovered = True  # prevent real fs access

        registry: Registry[str] = Registry(label="test", package=__name__)

        @registry.register("k")
        async def _f(s: Any) -> str:
            return "v"

        with patch.object(Registry, "_discover", _noop_discover):
            await registry.build("k", _fake_settings())

        assert call_count == 1

    async def test_discover_only_runs_once_across_many_builds(self) -> None:
        """_discover() must not be called again after the first build()."""
        call_count = 0

        def _noop_discover(self: Any) -> None:
            nonlocal call_count
            call_count += 1
            self._discovered = True

        registry: Registry[str] = Registry(label="test", package=__name__)

        @registry.register("item")
        async def _fac(s: Any) -> str:
            return "x"

        with patch.object(Registry, "_discover", _noop_discover):
            for _ in range(5):
                await registry.build("item", _fake_settings())

        assert call_count == 1


# ---------------------------------------------------------------------------
# Auto-discovery via pkgutil + importlib
# ---------------------------------------------------------------------------


class TestAutoDiscovery:
    def test_discover_imports_each_non_package_module(self) -> None:
        """_discover() must import every leaf module in the configured package."""
        registry: Registry[str] = Registry(label="scan", package="some.pkg")

        fake_pkg = types.SimpleNamespace(__path__=["/fake/path"])

        fake_modules = [
            (MagicMock(), "impl_a", False),
            (MagicMock(), "impl_b", False),
        ]

        with (
            patch.object(_pkgutil, "iter_modules", return_value=fake_modules),
            patch.object(_importlib, "import_module") as mock_import,
        ):
            mock_import.side_effect = lambda name: fake_pkg if name == "some.pkg" else MagicMock()
            registry._discover()

        imported = [call.args[0] for call in mock_import.call_args_list]
        assert "some.pkg.impl_a" in imported
        assert "some.pkg.impl_b" in imported

    def test_discover_skips_sub_packages(self) -> None:
        """_discover() must not import entries where ispkg=True."""
        registry: Registry[str] = Registry(label="test", package="some.pkg")

        fake_pkg = types.SimpleNamespace(__path__=["/fake/path"])

        fake_modules = [
            (MagicMock(), "sub_pkg", True),  # ispkg — must be skipped
            (MagicMock(), "impl", False),  # module — must be imported
        ]

        with (
            patch.object(_pkgutil, "iter_modules", return_value=fake_modules),
            patch.object(_importlib, "import_module") as mock_import,
        ):
            mock_import.side_effect = lambda name: fake_pkg if name == "some.pkg" else MagicMock()
            registry._discover()

        imported = [call.args[0] for call in mock_import.call_args_list]
        assert "some.pkg.sub_pkg" not in imported
        assert "some.pkg.impl" in imported

    def test_discover_sets_discovered_flag(self) -> None:
        """_discover() must set _discovered to True after scanning."""
        registry: Registry[str] = Registry(label="test", package="some.pkg")

        fake_pkg = types.SimpleNamespace(__path__=["/fake/path"])

        with (
            patch.object(_pkgutil, "iter_modules", return_value=[]),
            patch.object(_importlib, "import_module", return_value=fake_pkg),
        ):
            registry._discover()

        assert registry._discovered is True

    def test_discover_uses_package_path_for_iter_modules(self) -> None:
        """_discover() must pass pkg.__path__ to pkgutil.iter_modules."""
        registry: Registry[str] = Registry(label="test", package="some.pkg")

        fake_path = ["/specific/path"]
        fake_pkg = types.SimpleNamespace(__path__=fake_path)

        with (
            patch.object(_importlib, "import_module", return_value=fake_pkg),
            patch.object(_pkgutil, "iter_modules", return_value=[]) as mock_iter,
        ):
            registry._discover()

        assert mock_iter.call_args_list[0].args[0] == fake_path
