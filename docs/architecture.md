# Deep Research Agent — Architecture Guide

## System Overview

The Deep Research Agent is a modular, extensible AI system that accepts a natural-language research query and produces a structured, cited report. It orchestrates a ReAct (Reasoning + Acting) loop via LangGraph, combining hybrid vector+BM25 retrieval, cross-encoder reranking, persistent memory, and web scraping to synthesize comprehensive research outcomes. The system is exposed as a FastAPI service designed for Railway deployment, with all external dependencies abstracted behind protocols for testability and extensibility.

---

## Component Map

| Component                   | Layer         | Role               | Key Responsibility                                        |
| --------------------------- | ------------- | ------------------ | --------------------------------------------------------- |
| **FastAPI App**             | API           | HTTP server        | Request/response handling, auth, rate limiting            |
| **LangGraph Graph**         | Orchestration | ReAct loop         | Coordinates nodes, manages state, routing, checkpointing  |
| **Memory Node**             | Agent         | Recall             | Fetches relevant prior memories for the query             |
| **Retrieval Node**          | Agent         | Search             | Invokes hybrid retriever and reranker                     |
| **Reason Node**             | Agent         | Reasoning          | Calls LLM, parses tool calls or final answers             |
| **Tool Node**               | Agent         | Execution          | Dispatches Firecrawl search/scrape tools                  |
| **Synthesis Node**          | Agent         | Finalization       | Produces final report, persists summary to memory         |
| **HybridRetriever**         | Retrieval     | Multi-modal search | Dense + BM25 + RRF fusion over Qdrant                     |
| **HuggingFaceEmbedder**     | Retrieval     | Dense vectors      | Generates embeddings via HF Inference API                 |
| **BM25Encoder**             | Retrieval     | Sparse vectors     | Tokenizes and encodes documents/queries as sparse vectors |
| **FlashRankReranker**       | Retrieval     | Ranking            | Cross-encoder reranking, top-n selection                  |
| **HuggingFaceClient (LLM)** | LLM           | Chat inference     | Routes requests to Qwen3-32B via Featherless AI           |
| **Mem0MemoryService**       | Memory        | Session memory     | Persistent cross-turn recall via Mem0 API                 |
| **FirecrawlSearchTool**     | Tools         | Web search         | MCP wrapper for Firecrawl search                          |
| **FirecrawlScrapeTool**     | Tools         | Web scrape         | MCP wrapper for Firecrawl URL scraping                    |
| **Qdrant Client**           | Vector DB     | Storage            | Native hybrid search backend                              |
| **Settings (Config)**       | Config        | Environment        | Pydantic-based config from env vars                       |
| **Logging**                 | Observability | Structured logs    | structlog JSON or colored console output                  |

---

## Data Flow

### Single Research Request: HTTP POST → Report

```
1. HTTP Request (POST /research)
   └─> ResearchQuery(query, session_id, max_iterations)

2. LangGraph Graph invocation (ainvoke)
   └─> AgentState initialization

3. memory_node
   └─> MemoryService.search(session_id, query)
   └─> state["memories"] ← [prior_context_1, prior_context_2, ...]

4. retrieve_node
   ├─> HybridRetriever.retrieve(query, top_k=10)
   │   ├─> HuggingFaceEmbedder.embed(query) [async]
   │   ├─> BM25Encoder.encode_query(query) [via asyncio.to_thread]
   │   └─> Qdrant query_points(prefetch=[sparse, dense], RRF fusion)
   │
   └─> FlashRankReranker.rerank(query, results) [via asyncio.to_thread]
       └─> state["search_results"] ← [SearchResult(...), ...]

5. reason_node (loop iteration 1..N until max_iterations or final_answer)
   ├─> Build context from state["search_results"], state["memories"], state["tool_results"]
   ├─> LLMClient.complete([system_msg, prior_messages, user_query])
   │   └─> HuggingFaceClient calls Featherless AI (Qwen3-32B)
   │
   ├─> Parse response for:
   │   ├─> Tool calls: <tool_call>{"tool": "...", "input": {...}}</tool_call>
   │   └─> Final answer: <final_answer>...</final_answer>
   │
   └─> state["messages"] += assistant_response
       state["tool_calls_pending"] ← [parsed_tool_calls...]

6. Router: should_use_tools?
   ├─> YES (tool_calls_pending not empty) → go to tool_node
   └─> NO → go to should_continue

7. tool_node (if tools were called)
   └─> For each tool_call:
       ├─> Dispatch FirecrawlSearchTool.execute(SearchInput) or
       │   FirecrawlScrapeTool.execute(ScrapeInput)
       ├─> MCP session management (lazy connect, reuse, cleanup)
       └─> Collect ToolResult(is_error, content/error)
   └─> state["tool_results"] += tool_results
       state["messages"] += tool_result_messages
       state["tool_calls_pending"] = []

8. Loop back to reason_node (if iteration_count < max_iterations)

9. should_continue predicate
   ├─> if final_report is not None → "synthesize"
   ├─> if iteration_count >= max_iterations → "synthesize"
   └─> else → "retrieve" (start next iteration)

10. synthesis_node
    ├─> Build final context from accumulated search_results, memories, tool_results
    ├─> LLMClient.complete([system: "synthesis assistant", user: prompt + context])
    ├─> Extract summary from response
    ├─> Build ResearchReport with citations
    ├─> MemoryService.add(session_id, summary) [persist for future queries]
    └─> state["final_report"] ← ResearchReport(...)

11. Graph END
    └─> HTTP Response ← ResearchReport (JSON)
```

### Key State Transitions

The `AgentState` flows through every node, accumulating results:

- **query, session_id, max_iterations**: Immutable input (set once)
- **iteration_count**: Increments on each reason_node execution
- **search_results**: Replaced on each retrieve_node; contains current best documents
- **memories**: Set once in memory_node; prior context
- **messages**: Accumulates via `operator.add` (append-only); full conversation history
- **tool_results**: Accumulates via `operator.add`; all prior tool executions
- **tool_calls_pending**: Set by reason_node, cleared by tool_node
- **final_report**: Populated only by synthesis_node; terminates the loop

---

## Protocol Contracts

Every subsystem's external dependencies are hidden behind a protocol. All I/O methods are `async`.

### LLMClient

```python
@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, messages: list[Message]) -> str:
        """
        Args:
            messages: Non-empty list of Message(role, content).
                     Roles: "system", "user", "assistant", "tool".

        Returns:
            The model's text response.

        Raises:
            ValueError: If messages is empty or model returns no content.
        """
        ...
```

**Implementation**: `HuggingFaceClient`

- Wraps `AsyncInferenceClient(provider="featherless-ai")` for LLM inference
- Converts `Message` objects to `{"role": ..., "content": ...}` dicts
- Calls `client.chat.completions.create(model, messages, max_tokens)`
- Validates response content is not None
- Logs latency and response length
- Receives the dedicated LLM-only `AsyncInferenceClient` created at app startup

---

### Retriever

```python
@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """
        Returns up to *top_k* documents relevant to *query*.

        Raises:
            ValueError: If top_k is not a positive integer.
        """
        ...
```

**Implementation**: `HybridRetriever`

- Parallelizes dense and sparse encoding via `asyncio.gather()`
- Sends single `query_points()` call to Qdrant with:
  - Two `Prefetch` branches: one for sparse (BM25), one for dense
  - `FusionQuery(fusion=RRF)` for Reciprocal Rank Fusion
- Extracts and returns `SearchResult` objects from `ScoredPoint` payloads
- Returns up to top_k results

**Concrete Dependencies**:

- `HuggingFaceEmbedder`: Dense embedding generation
- `BM25Encoder`: Sparse vector encoding
- `AsyncQdrantClient`: Vector store interface

---

### Reranker

```python
@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Returns *results* sorted by relevance to *query*, highest first.
        Returns [] when results is empty.
        """
        ...
```

**Implementation**: `FlashRankReranker`

- Wraps synchronous `flashrank.Ranker.rerank()` via `asyncio.to_thread()`
- Converts `SearchResult` objects to `{"id": i, "text": r.content}` format
- Builds `RerankRequest(query=..., passages=...)`
- Scores results and returns top-n (capped by `top_n` constructor arg)
- Preserves URL and metadata; updates score field
- Logs input/output counts and latency

---

### MemoryService

```python
@runtime_checkable
class MemoryService(Protocol):
    async def add(self, session_id: str, content: str) -> None:
        """Persist *content* under *session_id*."""
        ...

    async def search(self, session_id: str, query: str) -> list[str]:
        """
        Return stored memories relevant to *query* for *session_id*.
        Returns empty list when no matches.
        """
        ...
```

**Implementation**: `Mem0MemoryService`

- Wraps `AsyncMemoryClient` (Mem0 managed API)
- `add()`: Wraps content in `{"role": "user", "content": content}` list, calls `client.add(messages, user_id=session_id)`
- `search()`: Calls `client.search(query, user_id=session_id)`, extracts `memory` field from results
- Silent handling of missing `memory` keys (schema evolution guard)

---

### Tool

```python
@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier matching the MCP tool name."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary for LLM tool selection."""
        ...

    async def execute(self, tool_input: ToolInput) -> ToolResult:
        """
        Raises:
            TypeError: If tool_input is not the expected concrete type.
            ToolExecutionError: If MCP call fails.
        """
        ...
```

**Implementations**: `FirecrawlSearchTool`, `FirecrawlScrapeTool`

- Both inherit from `_FirecrawlBaseTool` (MCP session lifecycle management)
- Lazy connect on first `execute()` call via double-checked locking
- Reuse `ClientSession` across multiple executions
- Handle `CallToolResult.isError` flag
- Extract text content from MCP response

---

### Embedder

```python
@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]:
        """
        Return a dense embedding for *text*.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        ...
```

**Implementation**: `HuggingFaceEmbedder`

- Wraps `AsyncInferenceClient` (standard HF Inference API, no provider kwarg)
- Calls `client.feature_extraction(text, model=model_id)` to generate dense embeddings
- Handles multiple output shapes from HF API (pooled, batch, token-level)
- Validates output dimension matches `expected_dim` (configured as `QDRANT_VECTOR_SIZE`)
- Returns normalized 1-D float list
- Receives the dedicated embeddings-only `AsyncInferenceClient` created at app startup

---

## LangGraph Graph Structure

### Topology

```
START
  │
  ▼
 memory  ──────────────────────────────┐
  │                                    │
  ▼                                    │
retrieve ◄─────────────────────────────┤── (should_continue → retrieve)
  │                                    │
  ▼                                    │
reason                                 │
  │                                    │
  ▼ should_use_tools                   │
┌──────────────────┐                   │
│ "tools"          │ "retrieve"        │
▼                  ▼                   │
tools          check_continue ─────────┘
  │                 │
  │ (loop back)     │ "synthesize"
  └──► reason       ▼
              synthesize
                  │
                 END
```

### Nodes (async functions)

1. **memory_node** — Fetches prior memories, updates `state["memories"]`
2. **retrieve_node** — Hybrid search + rerank, updates `state["search_results"]`
3. **reason_node** — LLM reasoning, parses tool calls, updates `state["messages"]` and `state["tool_calls_pending"]`
4. **tool_node** — Executes pending tools, updates `state["tool_results"]` and `state["messages"]`
5. **synthesis_node** — Final LLM synthesis, builds `ResearchReport`, persists summary

### Conditional Edges

- **reason → check_continue** (via `should_use_tools`):
  - If `tool_calls_pending` not empty → "tools"
  - Else → "check_continue" (a pass-through noop node)

- **check_continue → retrieve or synthesize** (via `should_continue`):
  - If `final_report` is not None → "synthesize"
  - If `iteration_count >= max_iterations` → "synthesize"
  - Else → "retrieve" (start next loop iteration)

- **tools → reason** (unconditional loop)

### Termination Conditions

The graph terminates when the synthesis_node is reached and completes. Synthesis is triggered when:

1. The agent emitted `<final_answer>...</final_answer>` in reasoning (sets `final_report`)
2. Max iterations exhausted (hard cap prevents infinite loops)

---

## Hybrid Retrieval Pipeline

### Architecture

```
Query Input
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  │
Dense Encoding    Sparse Encoding        │
(HuggingFace)     (BM25Encoder)          │
    │                  │                  │
    └──────────────────┴──────────────────┘
                       │
                       ▼
              asyncio.gather() [parallel]
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼ dense_vec                 ▼ sparse_idx, sparse_val

    Qdrant query_points()
    ├─ Prefetch(dense_vec, using="dense", limit=top_k*2)
    ├─ Prefetch(sparse_idx/val, using="sparse", limit=top_k*2)
    └─ FusionQuery(fusion=RRF)
                       │
                       ▼
             Reciprocal Rank Fusion
                       │
                       ▼
              FlashRankReranker
              (cross-encoder scoring)
                       │
                       ▼
            SearchResult[] (top_n)
```

### Components

#### HuggingFaceEmbedder

- **Model**: `BAAI/bge-large-en-v1.5` (1024-dim by default)
- **API**: HuggingFace Inference API (`AsyncInferenceClient`)
- **Output shapes handled**:
  - `(dim,)` → used directly
  - `(1, dim)` → take first element
  - `(seq, dim)` → take CLS token (index 0)
  - `(1, seq, dim)` → take first batch CLS token
- **Validation**: Raises `ValueError` if dim doesn't match `qdrant_vector_size`

#### BM25Encoder

- **Fit once** on corpus of documents (vocabulary and IDF table)
- **Tokenization**: Lowercase, split on `\W+`, drop empty tokens
- **encode_document(text)**:
  - Computes raw TF counts per token
  - Normalizes by document length
  - Multiplies by IDF (from BM25Okapi corpus statistics)
  - Returns `(indices, values)` sparse vector
- **encode_query(text)**:
  - Per-token IDF weight (max across repeats)
  - Unknown terms silently excluded
  - Returns empty `([], [])` if no known terms

#### Qdrant query_points() Call

- **Prefetch branches** execute in parallel on Qdrant:
  - Sparse: BM25 vectors, limit=top_k\*2
  - Dense: cosine similarity, limit=top_k\*2
- **Fusion**: RRF (Reciprocal Rank Fusion)
  - Combines rank positions from both branches
  - Formula: `score = sum(1 / (rank + 60))` per result (Qdrant default constant)
  - Balances dense + sparse contributions

#### FlashRankReranker

- **Input**: Up to top_k\*2 candidates (post-RRF)
- **Scoring**: Cross-encoder model (tiny BERT variant)
- **Output**: Re-sorted by `FlashRank.rerank()`, top-n returned
- **Score update**: Replaces original Qdrant score with FlashRank score
- **Threading**: Synchronous ranker wrapped via `asyncio.to_thread()`

### Score Fusion Formula (RRF)

For each result appearing in both dense and sparse rankings:

```
rrf_score = (1 / (rank_dense + 60)) + (1 / (rank_sparse + 60))
```

Results appearing in only one ranking get a contribution from that ranking only. This soft-balances vector and BM25 without hard weighting.

---

## Memory Architecture

### Session Scoping

- **session_id**: Opaque string (user ID, conversation ID, or request ID)
- **Mem0 user_id**: Maps directly to `session_id`
- **Scope**: All memories for a session are stored and retrieved together; no global cross-session memory

### Lifecycle

1. **memory_node (startup)**: `search(session_id, query)` → retrieve prior context
2. **synthesis_node (shutdown)**: `add(session_id, summary)` → persist final answer for future queries

### Content Format

- **add()**: Wraps plain text in Mem0 message format: `[{"role": "user", "content": content}]`
- **search()**: Returns list of plain-text memory strings; Mem0 handles chunking and embedding internally

### Mem0 API Surface Used

- `AsyncMemoryClient.add(messages, user_id)` — persist memories
- `AsyncMemoryClient.search(query, user_id)` — retrieve memories
- Mem0 is responsible for embedding, storage, and retrieval internally

---

## Tool Layer — MCP Transport

### MCP (Model Context Protocol)

The agent communicates with Firecrawl via MCP, a standardized tool interface. Tools are not direct imports; they are RPC calls through an MCP session.

### Session Management Pattern

Both `FirecrawlSearchTool` and `FirecrawlScrapeTool` use the same lifecycle:

1. **Lazy connect** (first `execute()` call):
   - `streamable_http_client(mcp_url)` opens HTTP bidirectional stream to Firecrawl MCP server
   - `ClientSession(read, write)` wraps the transport
   - `session.initialize()` establishes the MCP contract
   - Resources stored in `AsyncExitStack` for cleanup

2. **Double-checked locking** (concurrency safety):

   ```python
   if self._session is not None:
       return self._session
   async with self._lock:
       if self._session is not None:
           return self._session
       # ... actually connect ...
   ```

   - Prevents redundant connection attempts under concurrent calls
   - Only one `_ensure_connected()` succeeds; others wait for its result

3. **Session reuse** (all subsequent calls):
   - `session.call_tool(tool_name, arguments={...})` returns `CallToolResult`
   - Result content extracted from `TextContent` blocks

4. **Cleanup** (`aclose()`):
   - `AsyncExitStack.aclose()` releases all resources
   - Sets `_session` and `_exit_stack` to None

### Tool Implementations

#### FirecrawlSearchTool

- **MCP tool name**: `firecrawl_search`
- **Input**: `SearchInput(query: str, limit: int = 5)`
- **Arguments passed to MCP**: `{"query": ..., "limit": ...}`
- **Output**: Markdown content (search results with snippets)
- **Error handling**: Checks `result.isError` flag

#### FirecrawlScrapeTool

- **MCP tool name**: `firecrawl_scrape`
- **Input**: `ScrapeInput(url: str, only_main_content: bool = True)`
- **Arguments passed to MCP**: `{"url": ..., "onlyMainContent": ...}`
- **Output**: Markdown content (page text)
- **Error handling**: Checks `result.isError` flag

### Input Validation

Both `SearchInput` and `ScrapeInput` are frozen dataclasses with `__post_init__` validation:

```python
@dataclass(frozen=True)
class SearchInput(ToolInput):
    query: str
    limit: int = 5

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must be non-empty, non-whitespace")
        if self.limit <= 0:
            raise ValueError(f"limit must be positive")
```

Validation happens before tool execution, allowing the agent (and API) to fail fast.

---

## Configuration

### Settings Class (pydantic-settings)

All configuration is loaded from environment variables into a singleton `Settings` instance via `get_settings()`.

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # Cached for process lifetime
```

### Environment Variables (by subsystem)

#### LLM (HuggingFace / Featherless AI)

- `HF_TOKEN` ← API key for HF Hub (required)
- `LLM_MODEL` ← Model ID (default: `Qwen/Qwen3-32B`)
- `LLM_MAX_TOKENS` ← Max generation length (default: `4096`)

**Note**: Two separate `AsyncInferenceClient` instances are created at startup:

1. **LLM client** — `AsyncInferenceClient(provider="featherless-ai", api_key=HF_TOKEN)`
   - Routes inference to Qwen3-32B via Featherless AI
   - Used by `HuggingFaceClient` for agent reasoning and synthesis

2. **Embeddings client** — `AsyncInferenceClient(api_key=HF_TOKEN)` (no provider)
   - Uses standard HuggingFace Inference API
   - Supports feature-extraction task for dense embeddings
   - Used by `HuggingFaceEmbedder` for query and document encoding

This separation exists because Featherless AI (a Qwen3 specialist provider) does not support the feature-extraction task required for generating embeddings. Both clients share the same `HF_TOKEN` but route to different endpoints.

#### Vector Store (Qdrant)

- `QDRANT_URL` ← Server URL (default: `http://localhost:6333`)
- `QDRANT_API_KEY` ← Optional API key (required for cloud deployments)
- `QDRANT_COLLECTION` ← Collection name (default: `research_chunks`)
- `QDRANT_VECTOR_SIZE` ← Embedding dimension (default: `1024`, must match embedding model)

#### Embedding

- `EMBEDDING_MODEL` ← HF model ID (default: `BAAI/bge-large-en-v1.5`)

#### Memory (Mem0)

- `MEM0_API_KEY` ← Mem0 API key (required)
- `MEM0_RETENTION_DAYS` ← Memory TTL (default: `30`)

#### Web Tools (Firecrawl)

- `FIRECRAWL_API_KEY` ← Firecrawl API key (required)
- `FIRECRAWL_MCP_URL` ← Base URL of Firecrawl MCP server (default: `https://mcp.firecrawl.dev`)

**Note**: At startup, the full MCP URL is constructed as:
```
{FIRECRAWL_MCP_URL.rstrip('/')}/{FIRECRAWL_API_KEY}/v2/mcp
```

For example: `https://mcp.firecrawl.dev/your-api-key/v2/mcp`

This URL points to the v2 Streamable HTTP endpoint of the Firecrawl remote MCP server.

#### FastAPI / Auth

- `SECRET_KEY` ← JWT signing key (required, ≥32 bytes)
- `JWT_ALGORITHM` ← Signing algorithm (default: `HS256`)
- `JWT_EXPIRY_MINUTES` ← Token expiry (default: `60`)

#### Retrieval Parameters

- `RETRIEVAL_TOP_K` ← Pre-rerank candidate count (default: `20`)
- `RETRIEVAL_RERANK_TOP_N` ← Final result count (default: `5`)

#### Observability

- `LOG_LEVEL` ← Minimum severity (default: `INFO`)
- `LOG_JSON` ← JSON output when true (default: `false`)

#### Application

- `APP_VERSION` ← Version string (default: `0.1.0`)
- `ENVIRONMENT` ← Deployment env: `dev|staging|prod` (default: `dev`)

#### LangSmith Tracing (Optional)

- `LANGCHAIN_TRACING_V2` ← Enable tracing (default: `false`)
- `LANGCHAIN_API_KEY` ← LangSmith API key (optional, required if tracing enabled)

### Secret Validation

At startup, `Settings._validate_secrets()` rejects:

- Missing required secrets (raises `ValueError`)
- Placeholder values: `changeme`, `your-key-here`, `xxx`, etc.
- `SECRET_KEY` shorter than 32 bytes

This fail-fast approach ensures no accidental exposure of placeholder values.

---

## Design Decisions and Rationale

### 1. Async-First

**Choice**: Every I/O method is `async`; event loop is never blocked.

**Rationale**:

- The agent spends 90%+ of time waiting for external APIs (LLM, Qdrant, Firecrawl, HF embeddings)
- Async allows multiple concurrent research sessions on a single Railway dyno
- Migration cost from sync→async is far higher than async→sync; start async from day one
- LangGraph's `ainvoke()` and `astream()` are async-native

**Implementation**:

- LLMClient, Retriever, Reranker, MemoryService, Tool protocols: all `async`
- Synchronous-only libraries (BM25Encoder, flashrank.Ranker) wrapped via `asyncio.to_thread()`
- No `time.sleep()` — use `await asyncio.sleep()`

---

### 2. Protocol-Based Abstraction

**Choice**: Every external dependency is hidden behind a protocol (Interface Segregation + Dependency Inversion).

**Rationale**:

- Swappable implementations without code changes (e.g., swap Mem0 → Zep, Qdrant → Weaviate, HF → Claude)
- Testability: mock protocols in unit tests without instantiating real external services
- Type safety: `isinstance(obj, Protocol)` at graph construction time catches misconfigurations early
- Clear contracts: protocol docstrings document method signatures, args, returns, exceptions

**Example**:

```python
# Business logic depends on abstraction, not implementation
def make_retrieval_node(retriever: Retriever, reranker: Reranker) -> NodeFn:
    # Any Retriever impl (HybridRetriever, DummyRetriever, etc.) works here
    async def retrieval_node(state: AgentState) -> dict[str, Any]:
        results = await retriever.retrieve(state["query"], top_k=10)
        ...
```

---

### 3. LangGraph + ReAct Loop

**Choice**: Orchestrate reasoning + tool use via LangGraph with persistent checkpointing.

**Rationale**:

- **Checkpointing**: LangGraph automatically saves state at each node; resumable if interrupted
- **Clear topology**: Nodes, edges, conditional routing are explicit graphs, not buried in loops
- **Cycles**: LangGraph enforces that cycles (reason → tools → reason) are explicit edges, preventing mistakes
- **Supervision deferred**: Start with a single agent; multi-agent supervisor pattern added only if evidence exists
- **Native state management**: LangGraph's `Annotated` fields with operators (e.g., `operator.add`) handle list accumulation naturally

---

### 4. Hybrid Dense+BM25 Retrieval

**Choice**: Parallel dense (BAAI embedding) + sparse (BM25) vectorization, fused by RRF.

**Rationale**:

- **15–30% accuracy gain** over vector-alone (empirically established in RAG literature)
- **Complementary strengths**: Dense catches semantic similarity, BM25 catches exact term matches
- **Qdrant-native fusion**: No client-side merging; RRF fusion happens server-side
- **Cost-effective**: No separate BM25 service; Qdrant's native sparse index
- **Parallelism**: Dense and sparse encoding run concurrently before Qdrant query

---

### 5. FlashRank Cross-Encoder Reranking

**Choice**: Post-retrieval reranking via cross-encoder (lightweight fine-tuned BERT).

**Rationale**:

- **Minimal friction**: Lightest-weight reranker available; no service call
- **Significant lift**: 5–10 point accuracy improvement in top-k results
- **Synergy**: Better reranking compensates for slightly lower recall from RRF
- **Threading**: Synchronous library wrapped via `asyncio.to_thread()` (non-blocking)

---

### 6. Mem0 for Persistent Memory

**Choice**: Mem0 managed service behind `MemoryService` protocol.

**Rationale**:

- **Swappability**: Protocol allows swap to Zep or custom backend
- **Simplicity**: Managed API (no local model, no storage setup)
- **Cross-session recall**: Summaries from prior queries inform future queries
- **Session scoping**: user_id/session_id boundaries prevent memory leakage

---

### 7. Firecrawl via MCP

**Choice**: Web content acquisition through MCP (Model Context Protocol).

**Rationale**:

- **Standardized interface**: MCP is transport-agnostic; easy to swap for other web crawlers
- **Double-checked locking**: Session reuse without redundant connections
- **Graceful degradation**: Tool failures don't crash the agent; errors are captured in state
- **Lazy connect**: No session overhead until tools are actually needed

---

### 8. FastAPI for HTTP Service

**Choice**: FastAPI with async routes and middleware-enforced auth.

**Rationale**:

- **Async-native**: Routes are `async`, non-blocking
- **Automatic docs**: OpenAPI/Swagger generation
- **Middleware**: Auth, rate limiting, request ID binding happen at transport layer
- **Production-proven**: Widely deployed on Railway and similar platforms

---

### 9. Deferred Complexity

**Choice**: Implement only what is essential; defer advanced patterns until evidence exists.

**Rationale**:

- No supervisor/orchestrator multi-agent pattern — single agent sufficient
- No GraphRAG or knowledge graph layer — hybrid retrieval sufficient for now
- No adaptive RAG query router — fixed pipeline sufficient
- No corrective RAG web fallback — synthesis happens regardless

**Principle**: Complexity is added in response to measured gaps, not speculative "future-proofing."

---

## Value Objects and DTOs

### Immutable Value Objects (frozen dataclasses)

These represent _data that flows through the system_, not _behavior_:

- **Message(role, content)**: Chat message with constrained role values
- **SearchResult(content, url, score, metadata)**: Retrieval result
- **ToolResult(is_error, content|error, metadata)**: Tool execution outcome
- **Citation(url, content_snippet, relevance_score)**: Source attribution
- **ResearchReport(query, summary, citations, session_id)**: Final agent output
- **ResearchQuery(query, session_id, max_iterations)**: Agent input

### Input DTOs (frozen dataclasses)

These form the _calling contract_ for protocols and are validated eagerly:

- **SearchInput(query, limit)**: Web search tool input
- **ScrapeInput(url, only_main_content)**: URL scrape tool input

All are frozen to prevent mutation after validation.

---

## Logging and Observability

### Structured Logging (structlog)

Every subsystem emits structured logs with contextual key-value pairs:

```python
_log.info(
    "retrieval_executed",
    num_retrieved=len(raw),
    num_reranked=len(reranked),
    latency_ms=round((time.perf_counter() - t0) * 1000, 1),
)
```

### Log Output Formats

- **Development** (`log_json=false`): Colored, human-readable console output
- **Production** (`log_json=true`): JSON-per-line, suitable for Railway log aggregation

### Context Binding (Request ID)

Request IDs are bound via `bind_contextvars()` in middleware and automatically included in all logs within that async context:

```python
from research_agent.logging import bind_contextvars, clear_contextvars

# In middleware:
bind_contextvars(request_id=generate_request_id())
# ... all logs in this request now include request_id ...
clear_contextvars()  # cleanup
```

### LangSmith Integration (Optional)

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set, all LangGraph nodes, LLM calls, and tool executions are traced to LangSmith for debugging and monitoring.

---

## Example: Full Request Lifecycle

**User sends:**

```json
POST /research
{
  "query": "What are the latest advances in retrieval-augmented generation?",
  "session_id": "user_alice_123"
}
```

**Graph execution:**

1. **Initialization**: `AgentState` populated with query, session_id, max_iterations=5, iteration_count=0, all collections empty except final_report=None

2. **memory_node**: Mem0 searches for memories under session_id. If Alice has run prior research, summaries are returned.

3. **retrieve_node** (iteration 1):
   - HF embedder generates 1024-d dense vector for the query
   - BM25Encoder (pre-fitted on document corpus) generates sparse vector
   - `asyncio.gather()` runs both in parallel
   - Qdrant `query_points()` fuses dense + sparse with RRF, returns top 20
   - FlashRank cross-encoder reranks, returns top 5
   - `state["search_results"]` ← 5 SearchResult objects

4. **reason_node** (iteration 1):
   - System prompt injected with retrieved snippets and prior memories
   - LLM (Qwen3-32B via Featherless AI) reasons about whether it needs more information
   - LLM output: `<tool_call>{"tool": "firecrawl_search", "input": {"query": "recent RAG techniques 2025"}}</tool_call>`
   - Parsed into `tool_calls_pending`
   - New assistant message appended to state["messages"]

5. **Router**: `should_use_tools` → tool_calls_pending is not empty → route to "tools"

6. **tool_node**:
   - Dispatches `firecrawl_search` via MCP (lazy-connect, reuse session)
   - Returns ToolResult with search markdown content
   - `state["tool_results"]` appends result
   - Tool response message appended to state["messages"]

7. **Loop back**: `iteration_count` → 1, `should_continue` checks: not final_report, not max_iterations → route to "retrieve"

8. **retrieve_node** (iteration 2):
   - New query context includes prior search results + tool results
   - Same retrieval + rerank pipeline

9. **reason_node** (iteration 2):
   - LLM now has even more context
   - Output: `<final_answer>RAG has evolved significantly with ...[comprehensive answer]...</final_answer>`
   - `state["final_report"]` set by graph (detected by router)

10. **Router**: `should_use_tools` → no tool_calls_pending → route to "check_continue"
    - `should_continue` → final_report is set → route to "synthesize"

11. **synthesis_node**:
    - Synthesize final comprehensive report using LLM
    - Extract citations from search_results
    - Persist summary to Mem0 for future queries
    - `state["final_report"]` ← ResearchReport object

12. **Graph END**

13. **HTTP response**:

```json
{
  "query": "What are the latest advances in retrieval-augmented generation?",
  "summary": "RAG has evolved ... [comprehensive synthesized answer] ...",
  "citations": [
    {
      "url": "https://arxiv.org/abs/2405.12345",
      "content_snippet": "Recent advances in RAG include...",
      "relevance_score": 0.92
    },
    ...
  ],
  "session_id": "user_alice_123"
}
```

---

## Testing Strategy

### Unit Tests

- Mock protocols at every boundary (Retriever, Reranker, LLMClient, MemoryService, Tool)
- Test individual node functions in isolation
- Test state transitions and routing predicates
- Test input validation in dataclasses and tool inputs

### Integration Tests

- Real Qdrant instance (via Docker)
- Real Mem0 instance (or mock its async interface)
- Mock LLM and Firecrawl (expensive/slow external services)

### Coverage

- Minimum **95%** line coverage via pytest
- All I/O paths tested (success + error branches)
- Edge cases: empty results, max iterations, tool failures

---

## Future Extensibility

### Adding a New Tool

1. Create `src/research_agent/tools/my_tool.py`
2. Implement `Tool` protocol (name, description, execute)
3. Register in agent graph's tools list
4. Add env vars to `config.py` and `.env.example`

### Adding a New Retriever

1. Create `src/research_agent/retrieval/my_retriever.py`
2. Implement `Retriever` protocol
3. Swap in via config or graph factory call (no code changes elsewhere)

### Adding a New Memory Backend

1. Create `src/research_agent/memory/my_backend.py`
2. Implement `MemoryService` protocol
3. Inject in graph factory; all agent code works unchanged

### Upgrading the Reasoning Loop

1. Modify `make_reason_node()` or add new conditional nodes
2. Update `AgentState` fields if new data is needed
3. Retest routing predicates and state accumulation

---

## Summary

The Deep Research Agent is a tightly integrated system built on three foundational principles:

1. **Protocol-based abstraction**: Every external dependency is swappable
2. **Async-first design**: Event loop is never blocked; multiple concurrent sessions fit in one process
3. **Incremental complexity**: Start simple; add advanced patterns only when evidence exists

The ReAct loop, hybrid retrieval, cross-encoder reranking, and persistent memory work together to produce accurate, cited research reports. The modular architecture ensures that each component can be tested, reasoned about, and upgraded independently.
