# Deep Research Agent — Module Reference

Complete API documentation for all public classes, functions, and dataclasses in the `research_agent` package.

---

## research_agent.agent

The agent subsystem orchestrates the ReAct loop using LangGraph. It defines shared state, node factories, and routing predicates.

### AgentState

Shared mutable state passed through every LangGraph node. LangGraph merges partial updates from each node back into the state. Fields annotated with `Annotated[list[...], operator.add]` accumulate (append-only); all others are last-write-wins.

**Type:** `TypedDict`

**Fields:**

| Name                 | Type                                        | Description                                                                                                         |
| -------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `query`              | `str`                                       | The raw research question supplied by the caller.                                                                   |
| `session_id`         | `str`                                       | Opaque identifier shared with the memory service for cross-turn recall.                                             |
| `max_iterations`     | `int`                                       | Hard cap on ReAct loop cycles — guards against runaway agents.                                                      |
| `iteration_count`    | `int`                                       | Number of reason→act cycles completed so far.                                                                       |
| `search_results`     | `list[SearchResult]`                        | Most-recent retrieval results; replaced on each retrieval step.                                                     |
| `memories`           | `list[str]`                                 | Relevant memories retrieved from the memory service at loop start.                                                  |
| `messages`           | `Annotated[list[Message], operator.add]`    | Full conversation history sent to the LLM. Uses `operator.add` so each node appends without knowing current length. |
| `tool_results`       | `Annotated[list[ToolResult], operator.add]` | Accumulated tool execution outcomes. `operator.add` appends.                                                        |
| `tool_calls_pending` | `list[dict[str, Any]]`                      | Structured tool-call dicts parsed from latest LLM response, waiting to be dispatched by the tool node.              |
| `final_report`       | `ResearchReport \| None`                    | Populated by the synthesis node when the agent is done. `None` while the loop is running.                           |

---

### make_memory_node

Returns a node that fetches relevant memories for the current query.

**Signature:**

```python
def make_memory_node(memory_service: MemoryService) -> NodeFn
```

**Parameters:**

| Name             | Type            | Description                                               |
| ---------------- | --------------- | --------------------------------------------------------- |
| `memory_service` | `MemoryService` | Memory backend implementing the `MemoryService` protocol. |

**Returns:** `NodeFn` — Async callable with signature `(AgentState) -> dict[str, Any]`.

**Behavior:**

- Reads: `state["session_id"]`, `state["query"]`
- Writes: `{"memories": list[str]}`
- Side effects: Calls `memory_service.search()`

---

### make_retrieval_node

Returns a node that retrieves and reranks documents for the current query.

**Signature:**

```python
def make_retrieval_node(retriever: Retriever, reranker: Reranker) -> NodeFn
```

**Parameters:**

| Name        | Type        | Description                                               |
| ----------- | ----------- | --------------------------------------------------------- |
| `retriever` | `Retriever` | Retrieval implementation (typically `HybridRetriever`).   |
| `reranker`  | `Reranker`  | Reranking implementation (typically `FlashRankReranker`). |

**Returns:** `NodeFn` — Async callable with signature `(AgentState) -> dict[str, Any]`.

**Behavior:**

- Reads: `state["query"]`
- Writes: `{"search_results": list[SearchResult]}`
- Retrieves up to 10 results, reranks them, and logs latency
- Side effects: Calls `retriever.retrieve()` and `reranker.rerank()`

---

### make_reason_node

Returns a node that calls the LLM and parses tool calls or a final answer.

**Signature:**

```python
def make_reason_node(llm: LLMClient) -> NodeFn
```

**Parameters:**

| Name  | Type        | Description                                       |
| ----- | ----------- | ------------------------------------------------- |
| `llm` | `LLMClient` | LLM client implementing the `LLMClient` protocol. |

**Returns:** `NodeFn` — Async callable with signature `(AgentState) -> dict[str, Any]`.

**Behavior:**

- Reads: `state["query"]`, `state["memories"]`, `state["search_results"]`, `state["tool_results"]`, `state["messages"]`
- Writes: `{"messages": list[Message], "tool_calls_pending": list[dict], "iteration_count": int}`
- Injects prior memories and search results into the system prompt
- Parses tool calls from LLM response using regex `<tool_call>...</tool_call>`
- Silently skips malformed tool calls
- Side effects: Calls `llm.complete()`

---

### make_tool_node

Returns a node that dispatches all pending tool calls and collects results.

**Signature:**

```python
def make_tool_node(tools: list[Tool]) -> NodeFn
```

**Parameters:**

| Name    | Type         | Description                               |
| ------- | ------------ | ----------------------------------------- |
| `tools` | `list[Tool]` | List of available tools, indexed by name. |

**Returns:** `NodeFn` — Async callable with signature `(AgentState) -> dict[str, Any]`.

**Behavior:**

- Reads: `state["tool_calls_pending"]`
- Writes: `{"tool_results": list[ToolResult], "tool_calls_pending": [], "messages": list[Message]}`
- Validates tool names; logs warnings for unknowns
- Coerces tool input dicts to typed `SearchInput` or `ScrapeInput` using `_build_tool_input()`
- Executes each tool and collects results
- Side effects: Calls `tool.execute()` for each pending tool call

---

### make_synthesis_node

Returns a node that synthesises a final report and persists it to memory.

**Signature:**

```python
def make_synthesis_node(llm: LLMClient, memory_service: MemoryService) -> NodeFn
```

**Parameters:**

| Name             | Type            | Description                     |
| ---------------- | --------------- | ------------------------------- |
| `llm`            | `LLMClient`     | LLM client for synthesis.       |
| `memory_service` | `MemoryService` | Memory backend for persistence. |

**Returns:** `NodeFn` — Async callable with signature `(AgentState) -> dict[str, Any]`.

**Behavior:**

- Reads: `state["query"]`, `state["search_results"]`, `state["memories"]`, `state["tool_results"]`, `state["session_id"]`
- Writes: `{"final_report": ResearchReport}`
- Synthesises a comprehensive answer via LLM
- Extracts citations from search results
- Stores summary in memory for future sessions
- Side effects: Calls `llm.complete()` and `memory_service.add()`

---

### should_continue

Routing predicate that decides whether to run another ReAct iteration or move to synthesis.

**Signature:**

```python
def should_continue(state: AgentState) -> Literal["retrieve", "synthesize"]
```

**Parameters:**

| Name    | Type         | Description          |
| ------- | ------------ | -------------------- |
| `state` | `AgentState` | Current agent state. |

**Returns:** `Literal["retrieve", "synthesize"]` — Route to next node.

**Logic:**

- Returns `"synthesize"` if `state["final_report"]` is not `None`
- Returns `"synthesize"` if `state["iteration_count"] >= state["max_iterations"]`
- Returns `"retrieve"` otherwise

---

### should_use_tools

Routing predicate that dispatches to tool node or loops back to retrieval.

**Signature:**

```python
def should_use_tools(state: AgentState) -> Literal["tools", "retrieve"]
```

**Parameters:**

| Name    | Type         | Description          |
| ------- | ------------ | -------------------- |
| `state` | `AgentState` | Current agent state. |

**Returns:** `Literal["tools", "retrieve"]` — Route to next node.

**Logic:**

- Returns `"tools"` if `state["tool_calls_pending"]` is non-empty
- Returns `"retrieve"` otherwise

---

### create_graph

Constructs and compiles the ReAct research agent graph.

**Signature:**

```python
def create_graph(
    *,
    retriever: Retriever,
    reranker: Reranker,
    memory_service: MemoryService,
    llm_client: LLMClient,
    tools: list[Tool],
) -> object  # CompiledGraph
```

**Parameters:**

| Name             | Type            | Description                      |
| ---------------- | --------------- | -------------------------------- |
| `retriever`      | `Retriever`     | Hybrid retrieval implementation. |
| `reranker`       | `Reranker`      | Reranking implementation.        |
| `memory_service` | `MemoryService` | Memory backend.                  |
| `llm_client`     | `LLMClient`     | LLM client.                      |
| `tools`          | `list[Tool]`    | List of available tools.         |

**Returns:** `CompiledGraph` (typed as `object` due to LangGraph version variability) — Ready for `ainvoke()` / `astream()`.

**Raises:**

- `TypeError` if any dependency does not satisfy its expected protocol.

**Topology:**

```
START
  ↓
memory
  ↓
retrieve ←─────────────────────────────────┐
  ↓                                        │
reason                                     │
  ↓ should_use_tools                       │
┌─────────────────┐                        │
│ "tools" │ "retrieve"                     │
↓         ↓                                 │
tools   check_continue ─────────────────────┘
↓         │
reason    │ "synthesize"
          ↓
       synthesize
          ↓
         END
```

---

## research_agent.retrieval

The retrieval subsystem handles document search and reranking. It includes protocol definitions, BM25 sparse encoding, dense embeddings, hybrid retrieval, and reranking.

### SearchResult

Immutable value object representing a single document returned by retrieval.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name       | Type             | Description                               |
| ---------- | ---------------- | ----------------------------------------- |
| `content`  | `str`            | The extracted text/markdown content.      |
| `url`      | `str`            | Source URL (may be empty).                |
| `score`    | `float`          | Relevance score (higher = more relevant). |
| `metadata` | `dict[str, str]` | Additional source metadata.               |

---

### Embedder (Protocol)

Converts text to a dense float vector.

**Signature:**

```python
@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...
```

**Method: embed**

| Parameter | Type  | Description             |
| --------- | ----- | ----------------------- |
| `text`    | `str` | Non-empty input string. |

**Returns:** `list[float]` — Dense embedding vector.

**Raises:**

- `ValueError` if _text_ is empty or whitespace-only.

---

### Retriever (Protocol)

Retrieves candidate documents for a query.

**Signature:**

```python
@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]: ...
```

**Method: retrieve**

| Parameter | Type  | Description                              |
| --------- | ----- | ---------------------------------------- |
| `query`   | `str` | Natural-language research query.         |
| `top_k`   | `int` | Maximum number of results (must be > 0). |

**Returns:** `list[SearchResult]` — Up to _top_k_ relevant documents.

**Raises:**

- `ValueError` if _top_k_ is not a positive integer.

---

### Reranker (Protocol)

Re-orders a list of `SearchResult` by relevance to a query.

**Signature:**

```python
@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...
```

**Method: rerank**

| Parameter | Type                 | Description                       |
| --------- | -------------------- | --------------------------------- |
| `query`   | `str`                | Original research query.          |
| `results` | `list[SearchResult]` | Candidates from hybrid retriever. |

**Returns:** `list[SearchResult]` — Results sorted by relevance descending. Returns `[]` when _results_ is empty.

---

### HybridRetriever

Retrieves documents via hybrid dense+BM25 search with Qdrant-native RRF fusion.

**Signature:**

```python
class HybridRetriever:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedder: Embedder,
        bm25_encoder: BM25Encoder,
    ) -> None: ...
```

**Constructor Parameters:**

| Name           | Type                | Description                                                                      |
| -------------- | ------------------- | -------------------------------------------------------------------------------- |
| `client`       | `AsyncQdrantClient` | Authenticated async Qdrant client.                                               |
| `collection`   | `str`               | Qdrant collection name (must have `"dense"` and `"sparse"` named vector spaces). |
| `embedder`     | `Embedder`          | Implements `Embedder` protocol; generates dense vectors.                         |
| `bm25_encoder` | `BM25Encoder`       | Fitted BM25 encoder; generates sparse vectors.                                   |

**Method: retrieve**

```python
async def retrieve(self, query: str, top_k: int) -> list[SearchResult]
```

**Parameters:**

| Name    | Type  | Description                              |
| ------- | ----- | ---------------------------------------- |
| `query` | `str` | Natural-language research query.         |
| `top_k` | `int` | Maximum number of results (must be > 0). |

**Returns:** `list[SearchResult]` — Hybrid-ranked results.

**Raises:**

- `ValueError` if _top_k_ ≤ 0.

**Behavior:**

- Runs dense embedding and BM25 encoding concurrently via `asyncio.gather()`
- Sends both vectors to Qdrant in a single `query_points()` call
- Uses Qdrant-native RRF (Reciprocal Rank Fusion) to combine dense and sparse ranks
- Converts Qdrant `ScoredPoint` payloads to `SearchResult` objects
- Logs retrieval metrics (count, latency)

---

### BM25Encoder

Converts text to BM25 sparse vectors using a fixed vocabulary.

**Signature:**

```python
class BM25Encoder:
    def __init__(self) -> None: ...
```

**Attributes:**

| Name      | Type               | Description                                      |
| --------- | ------------------ | ------------------------------------------------ |
| `_vocab`  | `dict[str, int]`   | Mapping from term to vocabulary index.           |
| `_idf`    | `dict[str, float]` | IDF scores for each term.                        |
| `_fitted` | `bool`             | Flag indicating whether `fit()` has been called. |

**Method: fit**

```python
def fit(self, texts: list[str]) -> None
```

**Parameters:**

| Name    | Type        | Description                             |
| ------- | ----------- | --------------------------------------- |
| `texts` | `list[str]` | Non-empty list of raw document strings. |

**Raises:**

- `ValueError` if _texts_ is empty.

**Behavior:**

- Tokenizes all documents
- Builds vocabulary and IDF table via `BM25Okapi`
- Fully replaces previous state on re-fit
- Logs corpus size and vocabulary size
- **Synchronous** — call via `asyncio.to_thread()` in async contexts

---

**Method: encode_document**

```python
def encode_document(self, text: str) -> tuple[list[int], list[float]]
```

**Parameters:**

| Name   | Type  | Description              |
| ------ | ----- | ------------------------ |
| `text` | `str` | Document text to encode. |

**Returns:** `(indices, values)` where `indices[i]` is the vocabulary index and `values[i]` is its IDF-weighted TF score.

**Raises:**

- `RuntimeError` if `fit()` has not been called.

**Behavior:**

- Tokenizes text
- Computes term frequency (TF) normalized by document length
- Multiplies by IDF for each term
- Silently excludes unknown terms
- **Synchronous**

---

**Method: encode_query**

```python
def encode_query(self, text: str) -> tuple[list[int], list[float]]
```

**Parameters:**

| Name   | Type  | Description           |
| ------ | ----- | --------------------- |
| `text` | `str` | Query text to encode. |

**Returns:** `(indices, values)` where each known query term is assigned its IDF weight (maximum across repeated occurrences). Returns `([], [])` when no query token appears in vocabulary.

**Raises:**

- `RuntimeError` if `fit()` has not been called.

**Behavior:**

- Tokenizes text
- Takes maximum IDF for repeated terms
- Silently excludes unknown terms
- **Synchronous**

---

### HuggingFaceEmbedder

Generates dense embeddings via the HuggingFace Inference API.

**Signature:**

```python
class HuggingFaceEmbedder:
    def __init__(
        self,
        client: AsyncInferenceClient,
        model: str,
        expected_dim: int,
    ) -> None: ...
```

**Constructor Parameters:**

| Name           | Type                   | Description                                                 |
| -------------- | ---------------------- | ----------------------------------------------------------- |
| `client`       | `AsyncInferenceClient` | Authenticated async HuggingFace client.                     |
| `model`        | `str`                  | HuggingFace model ID (e.g., `"BAAI/bge-large-en-v1.5"`).    |
| `expected_dim` | `int`                  | Expected embedding dimensionality; validated on every call. |

**Method: embed**

```python
async def embed(self, text: str) -> list[float]
```

**Parameters:**

| Name   | Type  | Description             |
| ------ | ----- | ----------------------- |
| `text` | `str` | Non-empty input string. |

**Returns:** `list[float]` of length `expected_dim`.

**Raises:**

- `ValueError` if _text_ is empty or whitespace-only.
- `ValueError` if API returns vector of wrong dimension.

**Behavior:**

- Validates input non-empty
- Calls HuggingFace `feature_extraction()` API
- Normalizes output shape via `_pool()` to handle:
  - `(dim,)` — already pooled, used directly
  - `(1, dim)` — batch of one, first element taken
  - `(seq, dim)` — token-level, CLS token (index 0) taken
  - `(1, seq, dim)` — batched token-level, first batch CLS token taken
- Validates dimension matches expected
- Returns as `list[float]`

---

### FlashRankReranker

Re-ranks `SearchResult` items using FlashRank.

**Signature:**

```python
class FlashRankReranker:
    def __init__(self, ranker: Ranker, top_n: int) -> None: ...
```

**Constructor Parameters:**

| Name     | Type               | Description                                          |
| -------- | ------------------ | ---------------------------------------------------- |
| `ranker` | `flashrank.Ranker` | Pre-initialised FlashRank ranker instance.           |
| `top_n`  | `int`              | Maximum number of results to return after reranking. |

**Method: rerank**

```python
async def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]
```

**Parameters:**

| Name      | Type                 | Description                        |
| --------- | -------------------- | ---------------------------------- |
| `query`   | `str`                | Original research query.           |
| `results` | `list[SearchResult]` | Candidates from `HybridRetriever`. |

**Returns:** `list[SearchResult]` — Top _n_ results with updated FlashRank scores, sorted descending. Returns `[]` when _results_ is empty.

**Behavior:**

- Returns immediately if _results_ is empty
- Wraps FlashRank (synchronous) in `asyncio.to_thread()` to avoid blocking event loop
- Creates `RerankRequest` from query and passage content
- Sorts results by FlashRank score, capped at `top_n`
- Returns new `SearchResult` instances with updated scores (original objects unchanged)
- Logs reranking metrics (input count, output count, latency)

---

### ensure_collection

Create the hybrid-search collection if it does not already exist.

**Signature:**

```python
async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    vector_size: int,
) -> None
```

**Parameters:**

| Name          | Type                | Description                                                       |
| ------------- | ------------------- | ----------------------------------------------------------------- |
| `client`      | `AsyncQdrantClient` | Authenticated async Qdrant client.                                |
| `name`        | `str`               | Collection name (from `Settings.qdrant_collection`).              |
| `vector_size` | `int`               | Dense vector dimensionality (from `Settings.qdrant_vector_size`). |

**Behavior:**

- **Idempotent** — calling multiple times with same arguments is safe
- Checks if collection exists via `client.collection_exists()`
- If exists, logs status and returns
- If missing, creates collection with:
  - `"dense"` — cosine-similarity dense vectors of dimension _vector_size_
  - `"sparse"` — BM25 native sparse vectors
- Logs creation status

---

## research_agent.memory

The memory subsystem provides cross-session persistent storage. It defines the protocol and a Mem0-backed implementation.

### MemoryService (Protocol)

Protocol for a session-scoped persistent memory store.

**Signature:**

```python
@runtime_checkable
class MemoryService(Protocol):
    async def add(self, session_id: str, content: str) -> None: ...
    async def search(self, session_id: str, query: str) -> list[str]: ...
```

**Method: add**

```python
async def add(self, session_id: str, content: str) -> None
```

**Parameters:**

| Name         | Type  | Description                                                             |
| ------------ | ----- | ----------------------------------------------------------------------- |
| `session_id` | `str` | Identifier for the session/user (must be non-empty).                    |
| `content`    | `str` | Plain-text string to store; implementation decides chunking/formatting. |

---

**Method: search**

```python
async def search(self, session_id: str, query: str) -> list[str]
```

**Parameters:**

| Name         | Type  | Description                                 |
| ------------ | ----- | ------------------------------------------- |
| `session_id` | `str` | Scopes search to memories for this session. |
| `query`      | `str` | Natural-language query for retrieval.       |

**Returns:** `list[str]` — Ordered list of memory strings, most relevant first. Returns `[]` when no memories match.

---

### Mem0MemoryService

Persistent memory service backed by the Mem0 managed API.

**Signature:**

```python
class Mem0MemoryService:
    def __init__(self, client: AsyncMemoryClient) -> None: ...
```

**Constructor Parameters:**

| Name     | Type                     | Description                                                                             |
| -------- | ------------------------ | --------------------------------------------------------------------------------------- |
| `client` | `mem0.AsyncMemoryClient` | Initialised async Mem0 client. Inject rather than construct internally for testability. |

**Method: add**

```python
async def add(self, session_id: str, content: str) -> None
```

**Parameters:**

| Name         | Type  | Description                |
| ------------ | ----- | -------------------------- |
| `session_id` | `str` | Maps to Mem0 `user_id`.    |
| `content`    | `str` | Plain-text memory content. |

**Behavior:**

- Wraps content in a single-element messages list: `[{"role": "user", "content": content}]`
- Calls `client.add()` with this format
- Logs operation completion and latency

---

**Method: search**

```python
async def search(self, session_id: str, query: str) -> list[str]
```

**Parameters:**

| Name         | Type  | Description                    |
| ------------ | ----- | ------------------------------ |
| `session_id` | `str` | Maps to Mem0 `user_id`.        |
| `query`      | `str` | Natural-language memory query. |

**Returns:** `list[str]` — Memory strings extracted from Mem0 results.

**Behavior:**

- Calls `client.search(query, user_id=session_id)`
- Mem0 returns list of result dicts
- Extracts `"memory"` field from each entry (silently skips entries missing the key)
- Logs search completion, hit count, and latency

---

## research_agent.tools

The tools subsystem wraps external services via MCP. It includes protocol definitions and Firecrawl implementations.

### ToolInput

Base class for all tool inputs.

**Type:** `@dataclass(frozen=True)`

**Notes:**

- Subclass this to define input contracts for new tools
- `Tool.execute()` accepts any `ToolInput` subclass
- Adding new tools never requires modifying the `Tool` protocol

---

### SearchInput

Input for a web-search tool invocation.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name    | Type  | Default  | Description                                       |
| ------- | ----- | -------- | ------------------------------------------------- |
| `query` | `str` | required | Search query (must be non-empty, non-whitespace). |
| `limit` | `int` | 5        | Maximum results to return (must be positive).     |

**Raises (post_init):**

- `ValueError` if _query_ is empty/whitespace-only
- `ValueError` if _limit_ ≤ 0

---

### ScrapeInput

Input for a single-URL scrape tool invocation.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name                | Type   | Default  | Description                                     |
| ------------------- | ------ | -------- | ----------------------------------------------- |
| `url`               | `str`  | required | HTTP/HTTPS URL with valid host.                 |
| `only_main_content` | `bool` | `True`   | Extract only main content (skip navbars, etc.). |

**Raises (post_init):**

- `ValueError` if _url_ is not an HTTP/HTTPS URL with valid host

---

### ToolResult

Immutable value object representing the outcome of a tool execution.

**Type:** `@dataclass(frozen=True)` (from `research_agent.models.research`)

**Fields:**

| Name       | Type             | Default  | Description                                        |
| ---------- | ---------------- | -------- | -------------------------------------------------- |
| `is_error` | `bool`           | required | Whether execution failed.                          |
| `content`  | `str \| None`    | `None`   | Success payload (populated when `is_error=False`). |
| `error`    | `str \| None`    | `None`   | Error message (populated when `is_error=True`).    |
| `metadata` | `dict[str, str]` | `{}`     | Additional result metadata.                        |

**Invariant:**

- Exactly one of `content` or `error` is populated
- `is_error` always reflects which field carries data

---

### ToolExecutionError

Exception raised when a tool's MCP call fails or the remote server returns an error.

**Type:** `Exception`

---

### Tool (Protocol)

Protocol for agent tools that wrap external services via MCP.

**Signature:**

```python
@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    async def execute(self, tool_input: ToolInput) -> ToolResult: ...
```

**Property: name**

**Returns:** `str` — Stable identifier matching MCP tool name.

---

**Property: description**

**Returns:** `str` — Human-readable summary used by the agent for tool selection.

---

**Method: execute**

```python
async def execute(self, tool_input: ToolInput) -> ToolResult
```

**Parameters:**

| Name         | Type        | Description                                                |
| ------------ | ----------- | ---------------------------------------------------------- |
| `tool_input` | `ToolInput` | Typed input subclass (e.g., `SearchInput`, `ScrapeInput`). |

**Returns:** `ToolResult` — Success or error outcome.

**Raises:**

- `TypeError` if _tool_input_ is not the expected concrete type for this tool
- `ToolExecutionError` if the underlying MCP call fails

---

### \_FirecrawlBaseTool

Manages the MCP session lifecycle shared by Firecrawl tools.

**Signature:**

```python
class _FirecrawlBaseTool:
    def __init__(self, mcp_url: str) -> None: ...
```

**Constructor Parameters:**

| Name      | Type  | Description                             |
| --------- | ----- | --------------------------------------- |
| `mcp_url` | `str` | URL of the Firecrawl remote MCP server. |

**Attributes:**

| Name          | Type                     | Description                                          |
| ------------- | ------------------------ | ---------------------------------------------------- |
| `_mcp_url`    | `str`                    | MCP server URL.                                      |
| `_session`    | `ClientSession \| None`  | Active MCP session (lazy-loaded).                    |
| `_lock`       | `asyncio.Lock`           | Prevents redundant connections under concurrent use. |
| `_exit_stack` | `AsyncExitStack \| None` | Manages session lifecycle.                           |

**Method: \_ensure_connected**

```python
async def _ensure_connected(self) -> ClientSession
```

**Returns:** `ClientSession` — Active MCP session (connects on first call if needed).

**Behavior:**

- Implements double-checked locking pattern for thread safety
- Returns immediately if already connected
- Lazily establishes connection on first `execute()` call
- Uses `AsyncExitStack` to manage HTTP and session cleanup
- Logs connection status
- Propagates any exceptions after cleaning up partial stack

---

**Method: aclose**

```python
async def aclose(self) -> None
```

**Behavior:**

- Closes MCP session and releases transport resources
- Idempotent — safe to call multiple times
- Protected by lock to prevent race conditions

---

### FirecrawlSearchTool

Wraps the `firecrawl_search` MCP tool exposed by the Firecrawl server.

**Type:** Inherits from `_FirecrawlBaseTool`

**Property: name**

**Returns:** `"firecrawl_search"`

---

**Property: description**

**Returns:** `"Search the web using Firecrawl and return full-page markdown content. Use this when you need recent information, broad topic coverage, or want to discover relevant URLs before scraping."`

---

**Method: execute**

```python
async def execute(self, tool_input: ToolInput) -> ToolResult
```

**Parameters:**

| Name         | Type        | Description                     |
| ------------ | ----------- | ------------------------------- |
| `tool_input` | `ToolInput` | Must be `SearchInput` instance. |

**Returns:** `ToolResult` — Markdown content on success, error message on failure.

**Raises:**

- `TypeError` if _tool_input_ is not `SearchInput`
- `ToolExecutionError` if MCP call raises an exception

**Behavior:**

- Lazily connects to MCP server if needed
- Calls `session.call_tool("firecrawl_search", arguments={...})`
- Extracts markdown content from MCP response
- Returns `is_error=True` if Firecrawl reports an error
- Logs completion latency
- Content is joined from all `TextContent` blocks in the response

---

### FirecrawlScrapeTool

Wraps the `firecrawl_scrape` MCP tool exposed by the Firecrawl server.

**Type:** Inherits from `_FirecrawlBaseTool`

**Property: name**

**Returns:** `"firecrawl_scrape"`

---

**Property: description**

**Returns:** `"Scrape a single URL using Firecrawl and return the page content as markdown. Use this when you have a specific URL to fetch and need clean, structured text without navigating to it manually."`

---

**Method: execute**

```python
async def execute(self, tool_input: ToolInput) -> ToolResult
```

**Parameters:**

| Name         | Type        | Description                     |
| ------------ | ----------- | ------------------------------- |
| `tool_input` | `ToolInput` | Must be `ScrapeInput` instance. |

**Returns:** `ToolResult` — Page markdown on success, error message on failure.

**Raises:**

- `TypeError` if _tool_input_ is not `ScrapeInput`
- `ToolExecutionError` if MCP call raises an exception

**Behavior:**

- Lazily connects to MCP server if needed
- Extracts URL domain for logging
- Calls `session.call_tool("firecrawl_scrape", arguments={...})`
- Maps `only_main_content` to Firecrawl's `onlyMainContent`
- Returns `is_error=True` if Firecrawl reports an error
- Logs completion latency and URL domain
- Content is joined from all `TextContent` blocks in the response

---

## research_agent.llm

The LLM subsystem provides chat-completion inference. It defines the protocol and a HuggingFace/Featherless AI implementation.

### Message

Immutable value object representing a single chat message.

**Type:** `@dataclass(frozen=True)` (from `research_agent.models.research`)

**Fields:**

| Name      | Type                                             | Description                                                         |
| --------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| `role`    | `Literal["system", "user", "assistant", "tool"]` | Message role. Constrained to four standard OpenAI-compatible roles. |
| `content` | `str`                                            | Message text.                                                       |

**Validation (post_init):**

- Raises `ValueError` if _role_ is not in `{"system", "user", "assistant", "tool"}`

**Notes:**

- `"tool"` role carries tool-call results back to the model in multi-turn tool-use conversations
- All instances are immutable (frozen dataclass)

---

### LLMClient (Protocol)

Protocol for a chat-style language model client.

**Signature:**

```python
@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...
```

**Method: complete**

```python
async def complete(self, messages: list[Message]) -> str
```

**Parameters:**

| Name       | Type            | Description                                     |
| ---------- | --------------- | ----------------------------------------------- |
| `messages` | `list[Message]` | Ordered conversation turns (must not be empty). |

**Returns:** `str` — Model's text response.

**Raises:**

- `ValueError` if _messages_ is empty
- `ValueError` if the model returns no content

---

### HuggingFaceClient

LLM client backed by HuggingFace Hub's async inference API via Featherless AI.

**Signature:**

```python
class HuggingFaceClient:
    def __init__(
        self,
        client: AsyncInferenceClient,
        model: str,
        max_tokens: int,
    ) -> None: ...
```

**Constructor Parameters:**

| Name         | Type                                   | Description                                                            |
| ------------ | -------------------------------------- | ---------------------------------------------------------------------- |
| `client`     | `huggingface_hub.AsyncInferenceClient` | Initialised async HuggingFace client with `provider="featherless-ai"`. |
| `model`      | `str`                                  | HuggingFace model ID (e.g., `"Qwen/Qwen3-32B"`).                       |
| `max_tokens` | `int`                                  | Maximum tokens to generate per completion (applied to all calls).      |

**Method: complete**

```python
async def complete(self, messages: list[Message]) -> str
```

**Parameters:**

| Name       | Type            | Description                                     |
| ---------- | --------------- | ----------------------------------------------- |
| `messages` | `list[Message]` | Ordered conversation turns (must not be empty). |

**Returns:** `str` — Model's text response.

**Raises:**

- `ValueError` if _messages_ is empty
- `ValueError` if the model returns no content

**Behavior:**

- Validates messages non-empty
- Converts `Message` objects to dicts: `{"role": "...", "content": "..."}`
- Calls HuggingFace Hub via `client.chat.completions.create()`
- Routes through Featherless AI provider
- Applies `max_tokens` limit set at construction
- Logs request/response details and latency
- Extracts first choice message content

---

## research_agent.models

Shared value objects used across subsystems.

### SearchResult

A single document/page returned by a retrieval or search operation.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name       | Type             | Description                               |
| ---------- | ---------------- | ----------------------------------------- |
| `content`  | `str`            | Extracted text/markdown content.          |
| `url`      | `str`            | Source URL (may be empty).                |
| `score`    | `float`          | Relevance score (higher = more relevant). |
| `metadata` | `dict[str, str]` | Additional source metadata.               |

---

### Message

An immutable value object representing a single chat message.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name      | Type                                             | Description   |
| --------- | ------------------------------------------------ | ------------- |
| `role`    | `Literal["system", "user", "assistant", "tool"]` | Message role. |
| `content` | `str`                                            | Message text. |

**Validation:** Raises `ValueError` if role not in valid set.

---

### ToolResult

Immutable value object representing the outcome of a tool execution.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name       | Type             | Default  | Description               |
| ---------- | ---------------- | -------- | ------------------------- |
| `is_error` | `bool`           | required | Whether execution failed. |
| `content`  | `str \| None`    | `None`   | Success payload.          |
| `error`    | `str \| None`    | `None`   | Error message.            |
| `metadata` | `dict[str, str]` | `{}`     | Additional metadata.      |

**Invariant:** Exactly one of `content` or `error` is populated.

---

### ResearchQuery

Input value object for a research request.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name             | Type  | Default  | Description                            |
| ---------------- | ----- | -------- | -------------------------------------- |
| `query`          | `str` | required | Research question.                     |
| `session_id`     | `str` | required | Session identifier for memory scoping. |
| `max_iterations` | `int` | 5        | Hard cap on ReAct loop cycles.         |

**Validation (post_init):**

- Raises `ValueError` if _query_ is empty or whitespace
- Raises `ValueError` if _session_id_ is empty
- Raises `ValueError` if _max_iterations_ < 1

---

### Citation

A single source cited in a research report.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name              | Type    | Description                           |
| ----------------- | ------- | ------------------------------------- |
| `url`             | `str`   | Source URL.                           |
| `content_snippet` | `str`   | Excerpt from the source.              |
| `relevance_score` | `float` | Relevance score assigned by reranker. |

---

### ResearchReport

The completed output of a research session.

**Type:** `@dataclass(frozen=True)`

**Fields:**

| Name         | Type             | Description                      |
| ------------ | ---------------- | -------------------------------- |
| `query`      | `str`            | Original research question.      |
| `summary`    | `str`            | Synthesised answer.              |
| `citations`  | `list[Citation]` | Source citations (may be empty). |
| `session_id` | `str`            | Session identifier.              |

**Validation (post_init):**

- Raises `ValueError` if _query_ is empty or whitespace
- Raises `ValueError` if _session_id_ is empty

---

## research_agent.logging

Structured logging configuration and helpers.

### configure_logging

Configure structlog for the whole application.

**Signature:**

```python
def configure_logging(log_level: str = "INFO", log_json: bool = False) -> None
```

**Parameters:**

| Name        | Type   | Default  | Description                                                                                                  |
| ----------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------ |
| `log_level` | `str`  | `"INFO"` | Minimum severity: DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive).                                  |
| `log_json`  | `bool` | `False`  | When `True`, render as JSON (production/Railway). When `False`, render coloured human-readable output (dev). |

**Behavior:**

- Must be called once before first log statement
- Safe to call multiple times (reconfigures pipeline)
- Shares processors regardless of format (context vars, timestamps, stack info)
- JSON format adds `JSONRenderer()` and `UnicodeDecoder()`
- Console format adds `ConsoleRenderer()` with colors if stderr is a TTY
- LLM completion mode: `cache_logger_on_first_use=False` so tests can reconfigure

---

### get_logger

Return a named `structlog.stdlib.BoundLogger`.

**Signature:**

```python
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger
```

**Parameters:**

| Name   | Type          | Description                                                          |
| ------ | ------------- | -------------------------------------------------------------------- |
| `name` | `str \| None` | Logger name, typically `__name__`. Pass `None` for anonymous logger. |

**Returns:** `structlog.stdlib.BoundLogger` — Bound logger accepting `info()`, `debug()`, `warning()`, `error()`, `exception()` calls with arbitrary keyword context.

**Usage:**

```python
from research_agent.logging import get_logger

log = get_logger(__name__)
log.info("retrieve_complete", num_results=20, latency_ms=111.5)
```

---

### bind_contextvars

Bind values to the async context for automatic inclusion in all log calls.

**Signature:**

```python
bind_contextvars(**kwargs: Any) -> None
```

**Behavior:**

- Async-safe via contextvars module
- Every log call in this coroutine automatically includes bound context
- Useful for request IDs in API middleware

---

### clear_contextvars

Clear all async context variables.

**Signature:**

```python
def clear_contextvars() -> None
```

**Behavior:**

- Call in middleware teardown or after logging a request
- Prevents context leakage between requests

---

## research_agent.config

Application settings loaded from environment variables.

### Settings

Validated, immutable application settings.

**Type:** `pydantic_settings.BaseSettings`

**Notes:**

- All fields loaded from environment variables (case-insensitive)
- Required secrets validated at startup — app refuses to start if missing or contain placeholder values
- No `os.getenv()` calls permitted elsewhere — always use `get_settings()`

**Fields:**

| Name                     | Type                                                       | Default                    | Description                                                          | Env Var                  |
| ------------------------ | ---------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------- | ------------------------ |
| `hf_token`               | `SecretStr`                                                | required                   | HuggingFace token for Featherless AI inference                       | `HF_TOKEN`               |
| `llm_model`              | `str`                                                      | `"Qwen/Qwen3-32B"`         | Model identifier via Featherless AI                                  | `LLM_MODEL`              |
| `llm_max_tokens`         | `int`                                                      | `4096`                     | Max tokens per LLM completion (> 0)                                  | `LLM_MAX_TOKENS`         |
| `qdrant_url`             | `str`                                                      | `"http://localhost:6333"`  | Qdrant instance URL                                                  | `QDRANT_URL`             |
| `qdrant_api_key`         | `SecretStr \| None`                                        | `None`                     | Qdrant API key (required for cloud, omit for local)                  | `QDRANT_API_KEY`         |
| `qdrant_collection`      | `str`                                                      | `"research_chunks"`        | Collection name for research chunks                                  | `QDRANT_COLLECTION`      |
| `qdrant_vector_size`     | `int`                                                      | `1024`                     | Embedding dimensionality (> 0, must match embedding model)           | `QDRANT_VECTOR_SIZE`     |
| `embedding_model`        | `str`                                                      | `"BAAI/bge-large-en-v1.5"` | HuggingFace model for dense embeddings                               | `EMBEDDING_MODEL`        |
| `mem0_api_key`           | `SecretStr`                                                | required                   | Mem0 API key for persistent memory                                   | `MEM0_API_KEY`           |
| `mem0_retention_days`    | `int`                                                      | `30`                       | Session data retention period (> 0)                                  | `MEM0_RETENTION_DAYS`    |
| `firecrawl_api_key`      | `SecretStr`                                                | required                   | Firecrawl API key for web content acquisition                        | `FIRECRAWL_API_KEY`      |
| `secret_key`             | `SecretStr`                                                | required                   | JWT signing key (≥32 bytes)                                          | `SECRET_KEY`             |
| `jwt_algorithm`          | `str`                                                      | `"HS256"`                  | JWT algorithm                                                        | `JWT_ALGORITHM`          |
| `jwt_expiry_minutes`     | `int`                                                      | `60`                       | Token expiry (> 0)                                                   | `JWT_EXPIRY_MINUTES`     |
| `retrieval_top_k`        | `int`                                                      | `20`                       | Candidates before reranking (> 0)                                    | `RETRIEVAL_TOP_K`        |
| `retrieval_rerank_top_n` | `int`                                                      | `5`                        | Final results after reranking (> 0)                                  | `RETRIEVAL_RERANK_TOP_N` |
| `log_level`              | `Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]` | `"INFO"`                   | Logging verbosity                                                    | `LOG_LEVEL`              |
| `log_json`               | `bool`                                                     | `False`                    | Structured JSON logs (production/Railway)                            | `LOG_JSON`               |
| `app_version`            | `str`                                                      | `"0.1.0"`                  | Application version                                                  | `APP_VERSION`            |
| `environment`            | `Literal["dev", "staging", "prod"]`                        | `"dev"`                    | Deployment environment                                               | `ENVIRONMENT`            |
| `langchain_tracing_v2`   | `bool`                                                     | `False`                    | Enable LangSmith tracing (set `LANGCHAIN_TRACING_V2=true` to enable) | `LANGCHAIN_TRACING_V2`   |
| `langchain_api_key`      | `SecretStr \| None`                                        | `None`                     | LangSmith API key (required if `langchain_tracing_v2=true`)          | `LANGCHAIN_API_KEY`      |

---

### Validation: \_validate_secrets

Model validator that rejects missing or placeholder secret values at startup.

**Behavior:**

- Runs in `mode="after"` so all fields are already set
- Checks required secrets: `hf_token`, `mem0_api_key`, `firecrawl_api_key`, `secret_key`
- Rejects placeholder values: `changeme`, `your-key-here`, `todo`, `xxx`, `replace-me`, `<token>`, etc.
- Validates `secret_key` is ≥32 bytes
- Raises `ValueError` with clear message on any violation

---

### get_settings

Return the cached application settings instance.

**Signature:**

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings
```

**Returns:** `Settings` — Constructed once on first call, cached for process lifetime.

**Usage:**

```python
from research_agent.config import get_settings

settings = get_settings()
print(settings.qdrant_url)
```

**Testing Notes:**

- Call `get_settings.cache_clear()` after patching environment variables to force re-initialisation
- Caching prevents re-reading `.env` file on every call

---

## Cross-Cutting Patterns

### Immutability

All value objects (models, dataclasses) are frozen (`@dataclass(frozen=True)`) to prevent accidental mutation and enable safe concurrent access.

### Async-First

All I/O protocols are async:

- `Embedder.embed()` → `async`
- `Retriever.retrieve()` → `async`
- `Reranker.rerank()` → `async`
- `MemoryService.add()`, `search()` → `async`
- `Tool.execute()` → `async`
- `LLMClient.complete()` → `async`
- LangGraph nodes are `async def`

Synchronous operations (BM25, FlashRank) are wrapped in `asyncio.to_thread()` to prevent event loop blocking.

### Protocol-Driven Design

Every external dependency is hidden behind a protocol or abstract interface:

- Business logic depends on protocols, never concrete implementations
- Concrete implementations are injected at construction time
- Enables easy testing via mocks and swapping backends

### Fail Fast

- All inputs validated at boundaries (API, tool input dataclasses)
- Pydantic dataclass `__post_init__` validates constraints
- Exceptions raised immediately on violation, not returned as `None` or empty containers

### Structured Logging

- All modules use `structlog` via `get_logger()`
- Context variables bind request IDs automatically
- Logs include operation name, metrics (latency, count), and diagnostic info
- No `print()` statements in code

---

## Dependency Injection Conventions

The agent graph construction (`create_graph()`) accepts all dependencies as keyword-only arguments:

```python
graph = create_graph(
    retriever=hybrid_retriever,
    reranker=reranker,
    memory_service=memory,
    llm_client=llm,
    tools=[search_tool, scrape_tool],
)
```

Node factories (`make_*_node()`) close over injected dependencies:

```python
memory_node = make_memory_node(memory_service)
retrieval_node = make_retrieval_node(retriever, reranker)
reason_node = make_reason_node(llm_client)
tool_node = make_tool_node(tools)
synthesis_node = make_synthesis_node(llm_client, memory_service)
```

This is the idiomatic alternative to `functools.partial` when closures capture multiple objects.
