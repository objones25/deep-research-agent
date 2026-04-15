"""Unit tests for research_agent.api.container.

build_agent_runner delegates all subsystem construction to registries,
looked up by the provider/type keys in Settings.  Tests patch registry
singletons via ``patch.object`` so mocks intercept calls regardless of
which module holds a reference to the singleton.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.config import Settings
from research_agent.llm.registry import llm_registry
from research_agent.memory.registry import memory_registry
from research_agent.retrieval.registry import reranker_registry, retriever_registry
from research_agent.tools.registry import tools_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ENV: dict[str, str] = {
    "HF_TOKEN": "hf_" + "a" * 36,
    "QDRANT_API_KEY": "qdrant-key-abc123",
    "MEM0_API_KEY": "mem0-key-abc123",
    "FIRECRAWL_API_KEY": "firecrawl-key-abc123",
    "SECRET_KEY": "a" * 64,
}


def _make_settings(**kwargs: object) -> Settings:
    """Construct a Settings instance from _VALID_ENV, applying keyword overrides."""
    env = {k.lower(): v for k, v in _VALID_ENV.items()}
    env.update(kwargs)
    return Settings(_env_file=None, **env)


def _mock_llm_client() -> MagicMock:
    m = MagicMock()
    m.complete = AsyncMock()
    return m


def _mock_retriever() -> MagicMock:
    m = MagicMock()
    m.retrieve = AsyncMock()
    return m


def _mock_reranker() -> MagicMock:
    m = MagicMock()
    m.rerank = AsyncMock()
    return m


def _mock_memory_service() -> MagicMock:
    m = MagicMock()
    m.search = AsyncMock()
    m.add = AsyncMock()
    return m


def _all_registry_patches(
    *,
    llm_return: object = None,
    memory_return: object = None,
    retriever_return: object = None,
    reranker_return: object = None,
    tools_return: tuple = (),
) -> list:
    """Return a list of patch.object context managers for all five registries."""
    return [
        patch.object(
            llm_registry,
            "build",
            new_callable=AsyncMock,
            return_value=llm_return or _mock_llm_client(),
        ),
        patch.object(
            memory_registry,
            "build",
            new_callable=AsyncMock,
            return_value=memory_return or _mock_memory_service(),
        ),
        patch.object(
            retriever_registry,
            "build",
            new_callable=AsyncMock,
            return_value=retriever_return or _mock_retriever(),
        ),
        patch.object(
            reranker_registry,
            "build",
            new_callable=AsyncMock,
            return_value=reranker_return or _mock_reranker(),
        ),
        patch.object(
            tools_registry,
            "build",
            new_callable=AsyncMock,
            return_value=tools_return,
        ),
    ]


# ---------------------------------------------------------------------------
# TestBuildAgentRunner
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildAgentRunner:
    """build_agent_runner uses registries to build all dependencies."""

    async def test_returns_compiled_graph_runner(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_runner = MagicMock()

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", return_value=MagicMock()),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=mock_runner),
        ):
            result = await build_agent_runner(settings)

        assert result is mock_runner

    async def test_llm_registry_called_with_provider_from_settings(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_build_llm = AsyncMock(return_value=_mock_llm_client())

        with (
            patch.object(llm_registry, "build", mock_build_llm),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", return_value=MagicMock()),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        mock_build_llm.assert_awaited_once_with(settings.llm_provider, settings)

    async def test_memory_registry_called_with_provider_from_settings(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_build_memory = AsyncMock(return_value=_mock_memory_service())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(memory_registry, "build", mock_build_memory),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", return_value=MagicMock()),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        mock_build_memory.assert_awaited_once_with(settings.memory_provider, settings)

    async def test_retriever_registry_called_with_type_from_settings(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_build_retriever = AsyncMock(return_value=_mock_retriever())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(retriever_registry, "build", mock_build_retriever),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", return_value=MagicMock()),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        mock_build_retriever.assert_awaited_once_with(settings.retriever_type, settings)

    async def test_reranker_registry_called_with_type_from_settings(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_build_reranker = AsyncMock(return_value=_mock_reranker())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(reranker_registry, "build", mock_build_reranker),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", return_value=MagicMock()),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        mock_build_reranker.assert_awaited_once_with(settings.reranker_type, settings)

    async def test_tools_registry_called_for_each_enabled_tool(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_build_tools = AsyncMock(return_value=())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", mock_build_tools),
            patch("research_agent.agent.graph.create_graph", return_value=MagicMock()),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        for key in settings.enabled_tools:
            mock_build_tools.assert_any_await(key, settings)

    async def test_tool_results_are_aggregated_into_flat_tuple(self) -> None:
        from research_agent.agent.dependencies import AgentDependencies
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_tool_a = MagicMock()
        mock_tool_b = MagicMock()
        mock_create_graph = MagicMock(return_value=MagicMock())

        # tools_registry.build returns two tools for "firecrawl"
        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(
                tools_registry,
                "build",
                new_callable=AsyncMock,
                return_value=(mock_tool_a, mock_tool_b),
            ),
            patch("research_agent.agent.graph.create_graph", mock_create_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        args, _ = mock_create_graph.call_args
        deps: AgentDependencies = args[0]
        assert mock_tool_a in deps.tools
        assert mock_tool_b in deps.tools

    async def test_create_graph_receives_agent_dependencies(self) -> None:
        from research_agent.agent.dependencies import AgentDependencies
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_create_graph = MagicMock(return_value=MagicMock())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", mock_create_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        args, _ = mock_create_graph.call_args
        assert isinstance(args[0], AgentDependencies)

    async def test_passes_graph_to_compiled_runner(self) -> None:
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_graph = MagicMock()
        mock_runner_cls = MagicMock(return_value=MagicMock())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", return_value=mock_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", mock_runner_cls),
        ):
            await build_agent_runner(settings)

        mock_runner_cls.assert_called_once_with(mock_graph)

    async def test_agent_dependencies_receives_correct_llm_client(self) -> None:
        from research_agent.agent.dependencies import AgentDependencies
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_llm = _mock_llm_client()
        mock_create_graph = MagicMock(return_value=MagicMock())

        with (
            patch.object(llm_registry, "build", new_callable=AsyncMock, return_value=mock_llm),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", mock_create_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        args, _ = mock_create_graph.call_args
        deps: AgentDependencies = args[0]
        assert deps.llm_client is mock_llm

    async def test_agent_dependencies_receives_correct_memory_service(self) -> None:
        from research_agent.agent.dependencies import AgentDependencies
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_memory = _mock_memory_service()
        mock_create_graph = MagicMock(return_value=MagicMock())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry, "build", new_callable=AsyncMock, return_value=mock_memory
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", mock_create_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        args, _ = mock_create_graph.call_args
        deps: AgentDependencies = args[0]
        assert deps.memory_service is mock_memory

    async def test_agent_dependencies_receives_correct_retriever(self) -> None:
        from research_agent.agent.dependencies import AgentDependencies
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_ret = _mock_retriever()
        mock_create_graph = MagicMock(return_value=MagicMock())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=mock_ret
            ),
            patch.object(
                reranker_registry, "build", new_callable=AsyncMock, return_value=_mock_reranker()
            ),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", mock_create_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        args, _ = mock_create_graph.call_args
        deps: AgentDependencies = args[0]
        assert deps.retriever is mock_ret

    async def test_agent_dependencies_receives_correct_reranker(self) -> None:
        from research_agent.agent.dependencies import AgentDependencies
        from research_agent.api.container import build_agent_runner

        settings = _make_settings()
        mock_rer = _mock_reranker()
        mock_create_graph = MagicMock(return_value=MagicMock())

        with (
            patch.object(
                llm_registry, "build", new_callable=AsyncMock, return_value=_mock_llm_client()
            ),
            patch.object(
                memory_registry,
                "build",
                new_callable=AsyncMock,
                return_value=_mock_memory_service(),
            ),
            patch.object(
                retriever_registry, "build", new_callable=AsyncMock, return_value=_mock_retriever()
            ),
            patch.object(reranker_registry, "build", new_callable=AsyncMock, return_value=mock_rer),
            patch.object(tools_registry, "build", new_callable=AsyncMock, return_value=()),
            patch("research_agent.agent.graph.create_graph", mock_create_graph),
            patch("research_agent.agent.graph.CompiledGraphRunner", return_value=MagicMock()),
        ):
            await build_agent_runner(settings)

        args, _ = mock_create_graph.call_args
        deps: AgentDependencies = args[0]
        assert deps.reranker is mock_rer
