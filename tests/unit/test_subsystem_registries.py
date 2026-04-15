"""Unit tests for subsystem registry instances and factory registration.

Each subsystem (llm, memory, retrieval, tools) has a ``Registry[T]`` instance
in its own ``registry.py``.  The matching implementation modules register their
factories at module level via ``@registry.register("key")``.

These tests verify:
1. The registry module exports the expected ``Registry`` instance.
2. Importing the implementation module registers the expected key.
3. The factory, when called via ``build()``, returns an instance of the correct type.
4. The factory passes the right values from ``Settings`` to its dependencies.

All external I/O libraries are patched at their source-module attribute because
the factory functions use deferred ``from X import Y`` imports inside their bodies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _make_settings():  # type: ignore[return]
    from research_agent.config import Settings

    return Settings(_env_file=None, **{k.lower(): v for k, v in _VALID_ENV.items()})


# ---------------------------------------------------------------------------
# LLM registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMRegistry:
    """llm_registry is a Registry[LLMClient] and "huggingface" factory is registered."""

    def test_llm_registry_exists(self) -> None:
        from research_agent.llm.registry import llm_registry
        from research_agent.registry import Registry

        assert isinstance(llm_registry, Registry)

    def test_huggingface_key_registered_after_import(self) -> None:
        import research_agent.llm.huggingface  # noqa: F401 — triggers @register
        from research_agent.llm.registry import llm_registry

        assert "huggingface" in llm_registry._factories

    async def test_huggingface_factory_returns_hf_client(self) -> None:
        import research_agent.llm.huggingface  # noqa: F401
        from research_agent.llm.huggingface import HuggingFaceClient
        from research_agent.llm.registry import llm_registry

        settings = _make_settings()

        with patch("huggingface_hub.AsyncInferenceClient", return_value=MagicMock()):
            result = await llm_registry.build("huggingface", settings)

        assert isinstance(result, HuggingFaceClient)

    async def test_huggingface_factory_uses_featherless_provider(self) -> None:
        import research_agent.llm.huggingface  # noqa: F401
        from research_agent.llm.registry import llm_registry

        settings = _make_settings()
        call_kwargs: list[dict] = []

        def _capture(*args: object, **kwargs: object) -> MagicMock:
            call_kwargs.append(dict(kwargs))
            return MagicMock()

        with patch("huggingface_hub.AsyncInferenceClient", side_effect=_capture):
            await llm_registry.build("huggingface", settings)

        assert any(kw.get("provider") == "featherless-ai" for kw in call_kwargs)

    async def test_huggingface_factory_passes_model_from_settings(self) -> None:
        import research_agent.llm.huggingface  # noqa: F401
        from research_agent.llm.registry import llm_registry

        settings = _make_settings()
        mock_cls = MagicMock(return_value=MagicMock())

        with (
            patch("huggingface_hub.AsyncInferenceClient", return_value=MagicMock()),
            patch("research_agent.llm.huggingface.HuggingFaceClient", mock_cls),
        ):
            await llm_registry.build("huggingface", settings)

        _, called_kwargs = mock_cls.call_args
        assert called_kwargs["model"] == settings.llm_model

    async def test_huggingface_factory_passes_max_tokens_from_settings(self) -> None:
        import research_agent.llm.huggingface  # noqa: F401
        from research_agent.llm.registry import llm_registry

        settings = _make_settings()
        mock_cls = MagicMock(return_value=MagicMock())

        with (
            patch("huggingface_hub.AsyncInferenceClient", return_value=MagicMock()),
            patch("research_agent.llm.huggingface.HuggingFaceClient", mock_cls),
        ):
            await llm_registry.build("huggingface", settings)

        _, called_kwargs = mock_cls.call_args
        assert called_kwargs["max_tokens"] == settings.llm_max_tokens


# ---------------------------------------------------------------------------
# Memory registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryRegistry:
    """memory_registry is a Registry[MemoryService] and "mem0" factory is registered."""

    def test_memory_registry_exists(self) -> None:
        from research_agent.memory.registry import memory_registry
        from research_agent.registry import Registry

        assert isinstance(memory_registry, Registry)

    def test_mem0_key_registered_after_import(self) -> None:
        import research_agent.memory.mem0  # noqa: F401 — triggers @register
        from research_agent.memory.registry import memory_registry

        assert "mem0" in memory_registry._factories

    async def test_mem0_factory_returns_mem0_service(self) -> None:
        import research_agent.memory.mem0  # noqa: F401
        from research_agent.memory.mem0 import Mem0MemoryService
        from research_agent.memory.registry import memory_registry

        settings = _make_settings()

        with patch("mem0.AsyncMemoryClient", return_value=MagicMock()):
            result = await memory_registry.build("mem0", settings)

        assert isinstance(result, Mem0MemoryService)

    async def test_mem0_factory_passes_api_key(self) -> None:
        import research_agent.memory.mem0  # noqa: F401
        from research_agent.memory.registry import memory_registry

        settings = _make_settings()
        mock_client_cls = MagicMock(return_value=MagicMock())

        with patch("mem0.AsyncMemoryClient", mock_client_cls):
            await memory_registry.build("mem0", settings)

        _, called_kwargs = mock_client_cls.call_args
        assert called_kwargs["api_key"] == settings.mem0_api_key.get_secret_value()


# ---------------------------------------------------------------------------
# Retriever registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrieverRegistry:
    """retriever_registry is a Registry[Retriever] and "hybrid" factory is registered."""

    def test_retriever_registry_exists(self) -> None:
        from research_agent.registry import Registry
        from research_agent.retrieval.registry import retriever_registry

        assert isinstance(retriever_registry, Registry)

    def test_hybrid_key_registered_after_import(self) -> None:
        import research_agent.retrieval.hybrid  # noqa: F401 — triggers @register
        from research_agent.retrieval.registry import retriever_registry

        assert "hybrid" in retriever_registry._factories

    async def test_hybrid_factory_returns_hybrid_retriever(self) -> None:
        import research_agent.retrieval.hybrid  # noqa: F401
        from research_agent.retrieval.hybrid import HybridRetriever
        from research_agent.retrieval.registry import retriever_registry

        settings = _make_settings()

        with (
            patch("huggingface_hub.AsyncInferenceClient", return_value=MagicMock()),
            patch("qdrant_client.AsyncQdrantClient", return_value=MagicMock()),
            patch(
                "research_agent.retrieval.collection.ensure_collection",
                new_callable=AsyncMock,
            ),
            patch(
                "research_agent.retrieval.embedder.HuggingFaceEmbedder",
                return_value=MagicMock(),
            ),
            patch("research_agent.retrieval.bm25.BM25Encoder", return_value=MagicMock()),
        ):
            result = await retriever_registry.build("hybrid", settings)

        assert isinstance(result, HybridRetriever)

    async def test_hybrid_factory_passes_collection_from_settings(self) -> None:
        import research_agent.retrieval.hybrid  # noqa: F401
        from research_agent.retrieval.registry import retriever_registry

        settings = _make_settings()
        mock_retriever_cls = MagicMock(return_value=MagicMock())

        with (
            patch("huggingface_hub.AsyncInferenceClient", return_value=MagicMock()),
            patch("qdrant_client.AsyncQdrantClient", return_value=MagicMock()),
            patch(
                "research_agent.retrieval.collection.ensure_collection",
                new_callable=AsyncMock,
            ),
            patch(
                "research_agent.retrieval.embedder.HuggingFaceEmbedder",
                return_value=MagicMock(),
            ),
            patch("research_agent.retrieval.bm25.BM25Encoder", return_value=MagicMock()),
            patch("research_agent.retrieval.hybrid.HybridRetriever", mock_retriever_cls),
        ):
            await retriever_registry.build("hybrid", settings)

        _, called_kwargs = mock_retriever_cls.call_args
        assert called_kwargs["collection"] == settings.qdrant_collection

    async def test_hybrid_factory_passes_embedding_model_from_settings(self) -> None:
        import research_agent.retrieval.hybrid  # noqa: F401
        from research_agent.retrieval.registry import retriever_registry

        settings = _make_settings()
        mock_embedder_cls = MagicMock(return_value=MagicMock())

        with (
            patch("huggingface_hub.AsyncInferenceClient", return_value=MagicMock()),
            patch("qdrant_client.AsyncQdrantClient", return_value=MagicMock()),
            patch(
                "research_agent.retrieval.collection.ensure_collection",
                new_callable=AsyncMock,
            ),
            patch("research_agent.retrieval.embedder.HuggingFaceEmbedder", mock_embedder_cls),
            patch("research_agent.retrieval.bm25.BM25Encoder", return_value=MagicMock()),
            patch("research_agent.retrieval.hybrid.HybridRetriever", return_value=MagicMock()),
        ):
            await retriever_registry.build("hybrid", settings)

        _, called_kwargs = mock_embedder_cls.call_args
        assert called_kwargs["model"] == settings.embedding_model


# ---------------------------------------------------------------------------
# Reranker registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankerRegistry:
    """reranker_registry is a Registry[Reranker] and "flashrank" factory is registered."""

    def test_reranker_registry_exists(self) -> None:
        from research_agent.registry import Registry
        from research_agent.retrieval.registry import reranker_registry

        assert isinstance(reranker_registry, Registry)

    def test_flashrank_key_registered_after_import(self) -> None:
        import research_agent.retrieval.reranker  # noqa: F401 — triggers @register
        from research_agent.retrieval.registry import reranker_registry

        assert "flashrank" in reranker_registry._factories

    async def test_flashrank_factory_returns_flashrank_reranker(self) -> None:
        import research_agent.retrieval.reranker  # noqa: F401
        from research_agent.retrieval.registry import reranker_registry
        from research_agent.retrieval.reranker import FlashRankReranker

        settings = _make_settings()
        mock_ranker = MagicMock()

        with (
            patch("flashrank.Ranker", return_value=mock_ranker),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_ranker),
        ):
            result = await reranker_registry.build("flashrank", settings)

        assert isinstance(result, FlashRankReranker)

    async def test_flashrank_factory_passes_top_n_from_settings(self) -> None:
        import research_agent.retrieval.reranker  # noqa: F401
        from research_agent.retrieval.registry import reranker_registry

        settings = _make_settings()
        mock_ranker = MagicMock()
        mock_reranker_cls = MagicMock(return_value=MagicMock())

        with (
            patch("flashrank.Ranker", return_value=mock_ranker),
            patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_ranker),
            patch("research_agent.retrieval.reranker.FlashRankReranker", mock_reranker_cls),
        ):
            await reranker_registry.build("flashrank", settings)

        _, called_kwargs = mock_reranker_cls.call_args
        assert called_kwargs["top_n"] == settings.retrieval_rerank_top_n

    async def test_flashrank_factory_uses_to_thread_for_ranker(self) -> None:
        import research_agent.retrieval.reranker  # noqa: F401
        from research_agent.retrieval.registry import reranker_registry

        settings = _make_settings()
        mock_to_thread = AsyncMock(return_value=MagicMock())

        with (
            patch("flashrank.Ranker"),
            patch("asyncio.to_thread", mock_to_thread),
            patch("research_agent.retrieval.reranker.FlashRankReranker", return_value=MagicMock()),
        ):
            await reranker_registry.build("flashrank", settings)

        mock_to_thread.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tools registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToolsRegistry:
    """tools_registry is a Registry[tuple[Tool, ...]] and "firecrawl" factory is registered."""

    def test_tools_registry_exists(self) -> None:
        from research_agent.registry import Registry
        from research_agent.tools.registry import tools_registry

        assert isinstance(tools_registry, Registry)

    def test_firecrawl_key_registered_after_import(self) -> None:
        import research_agent.tools.firecrawl  # noqa: F401 — triggers @register
        from research_agent.tools.registry import tools_registry

        assert "firecrawl" in tools_registry._factories

    async def test_firecrawl_factory_returns_tuple(self) -> None:
        import research_agent.tools.firecrawl  # noqa: F401
        from research_agent.tools.registry import tools_registry

        settings = _make_settings()

        with (
            patch("research_agent.tools.firecrawl.FirecrawlSearchTool", return_value=MagicMock()),
            patch("research_agent.tools.firecrawl.FirecrawlScrapeTool", return_value=MagicMock()),
        ):
            result = await tools_registry.build("firecrawl", settings)

        assert isinstance(result, tuple)

    async def test_firecrawl_factory_returns_two_tools(self) -> None:
        import research_agent.tools.firecrawl  # noqa: F401
        from research_agent.tools.registry import tools_registry

        settings = _make_settings()

        with (
            patch("research_agent.tools.firecrawl.FirecrawlSearchTool", return_value=MagicMock()),
            patch("research_agent.tools.firecrawl.FirecrawlScrapeTool", return_value=MagicMock()),
        ):
            result = await tools_registry.build("firecrawl", settings)

        assert len(result) == 2

    async def test_firecrawl_factory_passes_mcp_endpoint(self) -> None:
        import research_agent.tools.firecrawl  # noqa: F401
        from research_agent.tools.registry import tools_registry

        settings = _make_settings()
        mock_search_cls = MagicMock(return_value=MagicMock())

        with (
            patch("research_agent.tools.firecrawl.FirecrawlSearchTool", mock_search_cls),
            patch("research_agent.tools.firecrawl.FirecrawlScrapeTool", return_value=MagicMock()),
        ):
            await tools_registry.build("firecrawl", settings)

        _, called_kwargs = mock_search_cls.call_args
        assert called_kwargs["mcp_url"] == settings.firecrawl_mcp_endpoint
