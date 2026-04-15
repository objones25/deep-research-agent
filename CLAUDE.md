# Deep Research Agent — CLAUDE.md

This file is the authoritative guide for any AI assistant working in this codebase.
Read it fully before writing any code, proposing any changes, or answering any questions.

---

## Project Purpose

A modular, extensible deep research agent that accepts a research query and returns a
structured, cited report. It uses a ReAct loop orchestrated by LangGraph, hybrid
vector+BM25 retrieval over a Qdrant store, FlashRank reranking, Mem0 for persistent
memory, and Firecrawl (via MCP) for web content acquisition. Exposed as a FastAPI service
deployed to Railway.

The architecture is designed to be incrementally enhanced. Start simple. Add complexity
only when a concrete gap is observed in production.

---

## Architecture Decisions (and Why)

| Concern         | Choice                                                         | Rationale                                                                                                             |
| --------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Orchestration   | LangGraph (ReAct loop)                                         | Persistent checkpointing, cycles, clean state machine; supervisor pattern deferred                                    |
| LLM             | Qwen3-32B via Featherless AI (huggingface_hub InferenceClient) | Best open-source agentic reasoning; single HF_TOKEN, no separate vendor account                                       |
| Retrieval       | Hybrid search (Qdrant dense + BM25) + RRF + FlashRank          | 15–30% better accuracy than vector alone; lowest-friction reranker                                                    |
| Vector DB       | Qdrant (self-hosted Docker)                                    | Native hybrid search, clean Python client, easy local dev                                                             |
| Memory          | Mem0 behind `MemoryService` protocol                           | Simple API, cross-session, swappable to Zep later                                                                     |
| Web acquisition | Firecrawl via MCP                                              | Standardized tool interface; swappable                                                                                |
| Concurrency     | Async-first (asyncio)                                          | I/O-bound workload; correct multi-user handling from day one; migration cost sync→async is far higher than async→sync |
| API             | FastAPI                                                        | Async-native, automatic OpenAPI docs, production-proven                                                               |
| Deploy          | Railway                                                        | Simple container deploys, managed env vars, no K8s overhead at this stage                                             |
| Package mgmt    | uv                                                             | Fast, deterministic, lockfile-based                                                                                   |

**Deferred decisions (do not implement until there is evidence they are needed):**

- Supervisor/orchestrator multi-agent pattern
- GraphRAG / knowledge graph layer
- Adaptive RAG query router
- Corrective RAG web fallback

---

## Repository Layout

```
.
├── CLAUDE.md                  # ← you are here
├── SECURITY.md
├── .claude/
│   └── rules.md               # AI assistant coding rules
├── pyproject.toml             # uv-managed; single source of truth for deps + tool config
├── .env.example               # all required env vars documented here, no defaults with secrets
├── src/
│   └── research_agent/
│       ├── agent/
│       │   ├── graph.py       # LangGraph graph definition (nodes, edges, state)
│       │   ├── nodes.py       # Individual node implementations
│       │   └── state.py       # AgentState dataclass
│       ├── retrieval/
│       │   ├── protocols.py   # Retriever, Reranker protocols + dataclasses
│       │   ├── hybrid.py      # HybridRetriever (Qdrant dense + BM25 + RRF)
│       │   └── reranker.py    # FlashRankReranker
│       ├── memory/
│       │   ├── protocols.py   # MemoryService protocol
│       │   └── mem0.py        # Mem0MemoryService implementation
│       ├── tools/
│       │   ├── protocols.py   # Tool protocol
│       │   └── firecrawl.py   # FirecrawlTool (MCP wrapper)
│       ├── llm/
│       │   ├── protocols.py     # LLMClient protocol
│       │   └── huggingface.py   # HuggingFaceClient (featherless-ai via InferenceClient)
│       ├── models/            # Shared Pydantic/dataclass models (no business logic)
│       │   └── research.py    # ResearchQuery, ResearchReport, SearchResult, etc.
│       ├── api/
│       │   ├── main.py        # FastAPI app factory
│       │   ├── routes/
│       │   │   └── research.py
│       │   └── middleware.py  # Auth, rate limiting, request ID
│       └── config.py          # Settings via pydantic-settings; reads from env
├── tests/
│   ├── conftest.py            # shared fixtures, mock factories
│   ├── unit/                  # fast, no I/O, everything mocked at the boundary
│   └── integration/           # real Qdrant + Mem0, still no LLM calls
└── Dockerfile
```

---

## Core Protocols and Abstractions

Every external dependency is hidden behind a protocol or ABC. This is non-negotiable.
If you are writing code that calls an external library directly inside business logic,
you are doing it wrong — put it in an implementation class that satisfies a protocol.

### Pattern

```python
# protocols.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class SearchResult:
    content: str
    url: str
    score: float
    metadata: dict[str, str]

@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]: ...
```

```python
# hybrid.py
from qdrant_client import AsyncQdrantClient  # always the async client
from research_agent.retrieval.protocols import Retriever, SearchResult

class HybridRetriever:
    def __init__(self, qdrant_client: AsyncQdrantClient, collection: str) -> None:
        self._client = qdrant_client
        self._collection = collection

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        ...
```

Key protocols to maintain — **all I/O methods are `async`**:

- `LLMClient` — `async complete(messages) -> str`
- `Retriever` — `async retrieve(query, top_k) -> list[SearchResult]`
- `Reranker` — `async rerank(query, results) -> list[SearchResult]`
- `MemoryService` — `async add(session_id, content)`, `async search(session_id, query) -> list[str]`
- `Tool` — `name: str`, `description: str`, `async execute(input) -> ToolResult`

---

## Development Principles

### Test-Driven Development

Write the test first. Always. No exceptions.

1. Write a failing test that describes the desired behaviour
2. Write the minimum code to make it pass
3. Refactor

Coverage must never drop below **95%**. CI will fail if it does. Do not add `# pragma: no cover`
without a comment explaining why and approval in review.

Run tests:

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95
```

### SOLID

- **Single Responsibility:** Each class has one reason to change. `HybridRetriever` retrieves.
  It does not rerank, it does not log, it does not cache.
- **Open/Closed:** Add new retrievers by implementing `Retriever`. Do not modify `HybridRetriever`
  to handle a new retrieval strategy — add a new class.
- **Liskov Substitution:** Any `Retriever` implementation must be substitutable. If your
  implementation requires callers to know what concrete type they have, the abstraction is wrong.
- **Interface Segregation:** Keep protocols small. `Retriever` should not also be a `Reranker`.
- **Dependency Inversion:** High-level modules (agent nodes) depend on protocols, never on
  concrete implementations. Inject implementations via constructors.

### Fail Early

- Raise immediately when a precondition is violated. Do not return `None` or an empty list
  and let the caller figure it out.
- No fallback paths unless a fallback is explicitly specced (tracked in an issue/PR).
- If you find yourself writing `except Exception: return default`, stop and reconsider.
- Validate all inputs at the boundary (API layer, tool inputs). Pydantic is your friend here.

### Simplicity

- If you are adding conditional branches to handle multiple retrieval strategies in a single
  function, there is a more elegant design — probably a protocol implementation.
- Do not add caching, retries, or circuit breakers until there is a measured reason to.
- No feature flags, no A/B infrastructure, no "future-proof" abstractions for things that
  don't exist yet.

### Async-First

This codebase is async-first. Every protocol method that performs I/O is `async`. There
are no sync I/O operations in the hot path.

**Always use the async client variant:**

- `AsyncQdrantClient` — never `QdrantClient` in production code
- `AsyncInferenceClient` — never `InferenceClient` in production code
- Mem0's async interface where available; wrap in `asyncio.to_thread` only as a last
  resort and document why

**Never block the event loop.** If a library only offers a sync interface:

1. Check Context7 — there is often an async variant you haven't found yet
2. If genuinely sync-only, wrap with `asyncio.to_thread()` and leave a comment
3. Do not use `time.sleep` — use `await asyncio.sleep`
4. Do not use `requests` — use `httpx.AsyncClient`

**Intra-step parallelism** (e.g. dense + sparse retrieval before RRF fusion) uses
`asyncio.gather()`. Add it only when you have both operations implemented and tested
individually — not speculatively.

**LangGraph nodes are `async def`:**

```python
# WRONG
def retrieve_node(state: AgentState) -> dict:
    results = retriever.retrieve(state.query, top_k=10)  # blocks event loop
    ...

# RIGHT
async def retrieve_node(state: AgentState) -> dict:
    results = await retriever.retrieve(state.query, top_k=10)
    ...
```

Use `graph.ainvoke()` and `graph.astream()` — never the sync variants.

| Tool    | Purpose                                       | Config location  |
| ------- | --------------------------------------------- | ---------------- |
| `black` | Formatting (non-negotiable, no style debates) | `pyproject.toml` |
| `ruff`  | Linting + import sorting                      | `pyproject.toml` |
| `mypy`  | Static type checking (strict mode)            | `pyproject.toml` |

All three run in CI. All three must pass cleanly before merge.

```bash
uv run black src tests
uv run ruff check src tests --fix
uv run mypy src
```

### Type Annotations

Every function signature must be fully annotated. No `Any` unless interfacing with a
third-party library that genuinely returns `Any` — document why with a comment.

### Dataclasses over Dicts

Use `@dataclass` (or `@dataclass(frozen=True)` for value objects) everywhere internal
data is passed around. Do not pass raw `dict[str, Any]` between modules.

---

## Using Context7 for Documentation

**Never assume how a library works. Always verify.**

Before writing any code that uses a library you haven't used in this session:

```
use context7 to retrieve documentation for <library-name>
```

This applies to: `langgraph`, `qdrant-client`, `mem0ai`, `flashrank`, `langchain-*`,
`pydantic`, `pydantic-settings`, `fastapi`, and any other non-stdlib library.

If the documentation is unclear or the API surface you need isn't covered, say so.
Do not guess. Do not extrapolate from older versions. Do not mock a library before you
have verified its real interface.

---

## Environment Variables

All config lives in `src/research_agent/config.py` as a `pydantic-settings` `Settings` class.
No `os.getenv()` calls outside of that file. No hardcoded values anywhere.

Required variables (see `.env.example` for full list):

- `HF_TOKEN` — HuggingFace token; used by `InferenceClient(provider="featherless-ai")`
- `QDRANT_URL`, `QDRANT_API_KEY`
- `MEM0_API_KEY`
- `FIRECRAWL_API_KEY`
- `SECRET_KEY` (FastAPI auth)

---

## LangGraph Conventions

- Agent state is a single `@dataclass` (`AgentState` in `agent/state.py`). No mutable
  defaults — use `field(default_factory=...)`.
- Each node is an `async def` function: `async (AgentState) -> dict` (partial state
  update). No side effects outside of explicit I/O nodes.
- Edges that branch on conditions use typed `Literal` return values from the condition
  function — never bare strings.
- The graph is constructed once in `agent/graph.py` and compiled into a `CompiledGraph`.
  Do not reconstruct the graph per request.
- Always use `await graph.ainvoke(...)` and `async for chunk in graph.astream(...)`.
  The sync `invoke` and `stream` variants are forbidden — they block the event loop.

---

## API Conventions

- All routes are `async`.
- Request/response models are Pydantic `BaseModel`s in `api/routes/`.
- Errors are raised as `HTTPException` — never return error payloads with 200 status.
- The FastAPI app is created via a factory function `create_app()` to support testing.
- Authentication is enforced via middleware, not per-route decorators.

---

## Adding a New Tool

1. Create `src/research_agent/tools/<name>.py`
2. Implement the `Tool` protocol
3. Write unit tests first (mock the external service at the boundary)
4. Register it in the agent's tool list in `agent/graph.py`
5. Add required env vars to `config.py` and `.env.example`
6. Document it in `SECURITY.md` under external integrations

## Adding a New Retriever

1. Create `src/research_agent/retrieval/<name>.py`
2. Implement `Retriever` protocol
3. Write unit tests (mock Qdrant or whatever backend it uses)
4. Write an integration test that hits a real local instance
5. It can be swapped in via config — no code change to the agent required

---

## Running Locally

```bash
# Install dependencies
uv sync

# Start Qdrant
docker compose up qdrant -d

# Copy and fill env
cp .env.example .env

# Run the API
uv run fastapi dev src/research_agent/api/main.py

# Run tests
uv run pytest

# Full quality check
uv run black src tests && uv run ruff check src tests && uv run mypy src && uv run pytest --cov=src --cov-fail-under=95
```
