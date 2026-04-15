"""FastAPI application factory.

``create_app()`` is the sole entry point for constructing the FastAPI
application. It is called by:
  * The ``uvicorn`` / ``fastapi dev`` process (via ``main:app`` in
    ``pyproject.toml`` scripts or the Railway ``CMD``).
  * Test fixtures — each test that needs a running app calls ``create_app()``
    with a mock ``AgentRunner`` to keep tests isolated.

Wiring order:
  1. ``lifespan`` context manager — logging init, LangSmith activation,
     agent runner stored in ``app.state``.
  2. ``RequestIDMiddleware`` — must be added after the app is created.
  3. Route inclusion — ``/health`` (unauthenticated), ``/research`` (JWT + rate-limit).
  4. Exception handler — translate unhandled exceptions to 500 JSON responses.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from research_agent.api.dependencies import AgentRunner
from research_agent.api.middleware import RequestIDMiddleware
from research_agent.api.routes import health as health_module
from research_agent.api.routes import research as research_module
from research_agent.config import Settings, get_settings
from research_agent.logging import configure_logging


async def _build_agent_runner(settings: Settings) -> AgentRunner:
    """Construct all production dependencies and return a wired ``AgentRunner``.

    All imports are deferred into this function so that heavyweight libraries
    (flashrank model download, qdrant client, etc.) are only loaded when the
    app starts — not at import time during test collection.

    Args:
        settings: Validated application settings loaded from environment.

    Returns:
        A :class:`~research_agent.agent.graph.CompiledGraphRunner` ready to
        handle research queries.
    """
    # Deferred imports — keep at function scope to avoid import-time side effects
    from flashrank import Ranker
    from huggingface_hub import AsyncInferenceClient
    from mem0 import AsyncMemoryClient
    from qdrant_client import AsyncQdrantClient

    from research_agent.agent.graph import CompiledGraphRunner, create_graph
    from research_agent.llm.huggingface import HuggingFaceClient
    from research_agent.memory.mem0 import Mem0MemoryService
    from research_agent.retrieval.bm25 import BM25Encoder
    from research_agent.retrieval.collection import ensure_collection
    from research_agent.retrieval.embedder import HuggingFaceEmbedder
    from research_agent.retrieval.hybrid import HybridRetriever
    from research_agent.retrieval.reranker import FlashRankReranker
    from research_agent.tools.firecrawl import FirecrawlScrapeTool, FirecrawlSearchTool
    from research_agent.tools.protocols import Tool

    # ------------------------------------------------------------------
    # HuggingFace inference clients
    #
    # Two separate clients are required because Featherless AI only supports
    # conversational/text-generation tasks — it does NOT support
    # feature-extraction (embeddings).  The standard HF Inference API
    # (no provider kwarg) handles embeddings for public models.
    # ------------------------------------------------------------------
    hf_client = AsyncInferenceClient(
        provider="featherless-ai",
        api_key=settings.hf_token.get_secret_value(),
    )
    embed_client = AsyncInferenceClient(
        api_key=settings.hf_token.get_secret_value(),
    )

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    qdrant_kwargs: dict[str, object] = {"url": settings.qdrant_url}
    if settings.qdrant_api_key is not None:
        qdrant_kwargs["api_key"] = settings.qdrant_api_key.get_secret_value()
    qdrant_client = AsyncQdrantClient(**qdrant_kwargs)  # type: ignore[arg-type]

    await ensure_collection(
        client=qdrant_client,
        name=settings.qdrant_collection,
        vector_size=settings.qdrant_vector_size,
    )

    # ------------------------------------------------------------------
    # Retrieval components
    # ------------------------------------------------------------------
    embedder = HuggingFaceEmbedder(
        client=embed_client,
        model=settings.embedding_model,
        expected_dim=settings.qdrant_vector_size,
    )

    # Fit BM25 with a minimal bootstrap corpus so the encoder is ready.
    # Real documents are indexed at query time via the retrieval pipeline.
    bm25 = BM25Encoder()
    bm25.fit(["research agent bootstrap corpus placeholder document"])

    retriever = HybridRetriever(
        client=qdrant_client,
        collection=settings.qdrant_collection,
        embedder=embedder,
        bm25_encoder=bm25,
    )

    # FlashRank Ranker is synchronous; instantiation downloads the model on
    # first call — this happens once at startup here, not per request.
    import asyncio

    ranker: Ranker = await asyncio.to_thread(Ranker)
    reranker = FlashRankReranker(ranker=ranker, top_n=settings.retrieval_rerank_top_n)

    # ------------------------------------------------------------------
    # Memory service
    # ------------------------------------------------------------------
    memory_client = AsyncMemoryClient(api_key=settings.mem0_api_key.get_secret_value())
    memory_service = Mem0MemoryService(client=memory_client)

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------
    llm_client = HuggingFaceClient(
        client=hf_client,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )

    # ------------------------------------------------------------------
    # Firecrawl tools
    # ------------------------------------------------------------------
    # The Firecrawl remote MCP server authenticates via an API-key path segment.
    # The v2 Streamable HTTP endpoint is at /<api_key>/v2/mcp.
    mcp_url = (
        f"{settings.firecrawl_mcp_url.rstrip('/')}"
        f"/{settings.firecrawl_api_key.get_secret_value()}/v2/mcp"
    )
    tools: list[Tool] = [
        FirecrawlSearchTool(mcp_url=mcp_url),
        FirecrawlScrapeTool(mcp_url=mcp_url),
    ]

    # ------------------------------------------------------------------
    # Assemble graph and return runner
    # ------------------------------------------------------------------
    graph = create_graph(
        retriever=retriever,
        reranker=reranker,
        memory_service=memory_service,
        llm_client=llm_client,
        tools=tools,
    )
    return CompiledGraphRunner(graph)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: run startup logic, yield, then run shutdown logic."""
    settings = get_settings()

    # Configure structured logging
    configure_logging(log_level=settings.log_level, log_json=settings.log_json)

    # Activate LangSmith tracing when both env vars are configured
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()

    # Build the production agent runner when none was injected (i.e. not in tests)
    if app.state.agent_runner is None:
        app.state.agent_runner = await _build_agent_runner(settings)

    yield

    # Shutdown — nothing to teardown at this stage


def create_app(agent_runner: AgentRunner | None = None) -> FastAPI:
    """Construct and configure a FastAPI application.

    Args:
        agent_runner: An ``AgentRunner`` implementation injected at construction
            time (production passes the compiled LangGraph; tests pass a mock).
            Stored as ``app.state.agent_runner`` so routes can retrieve it via
            ``request.app.state.agent_runner``.

    Returns:
        A fully configured ``FastAPI`` instance ready to be served by uvicorn.
    """
    settings = get_settings()

    app = FastAPI(
        title="Deep Research Agent",
        description=(
            "A modular AI research agent that accepts a query and returns "
            "a structured, cited report using hybrid retrieval and LangGraph."
        ),
        version=settings.app_version,
        lifespan=_lifespan,
        # Disable docs in production
        docs_url=None if settings.environment == "prod" else "/docs",
        redoc_url=None if settings.environment == "prod" else "/redoc",
    )

    # Store the agent runner so route handlers can retrieve it from app.state
    app.state.agent_runner = agent_runner

    # Middleware (applied outermost-first in the call stack)
    app.add_middleware(RequestIDMiddleware)

    # Routes
    app.include_router(health_module.router)
    app.include_router(research_module.router)

    # Global exception handler — prevent stack traces from leaking to clients
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred."},
        )

    return app


# No module-level app instance. Use uvicorn's --factory flag:
#
#   uvicorn research_agent.api.main:create_app --factory
#
# This prevents get_settings() from running at import time, which would fail
# in test collection where env vars are not yet set by fixtures.
