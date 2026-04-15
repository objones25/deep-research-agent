# Deep Research Agent — Developer Guide

This guide covers everything you need to develop, test, and deploy the deep-research-agent project locally and on Railway.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup](#local-setup)
3. [Environment Variables](#environment-variables)
4. [Running the Test Suite](#running-the-test-suite)
5. [Linting and Formatting](#linting-and-formatting)
6. [Full Quality Gate](#full-quality-gate)
7. [CI/CD Pipelines](#cicd-pipelines)
8. [Test-Driven Development Workflow](#test-driven-development-workflow)
9. [Adding New Components](#adding-new-components)
10. [Project Layout](#project-layout)
11. [Dependency Management](#dependency-management)

---

## Prerequisites

### System Requirements

- **Python 3.12** or later (the project targets Python 3.12 with `pyproject.toml`)
- **Docker** (for running Qdrant locally)
- **uv** (Python package manager — ultra-fast, lockfile-based)
  - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or via Homebrew: `brew install uv`

### External Accounts

Before you can run the application fully, you'll need credentials for these services. You can set up a minimal development environment with placeholders, but full features require real keys:

| Service     | Purpose                          | Sign-up Link                                       |
| ----------- | -------------------------------- | -------------------------------------------------- |
| HuggingFace | LLM inference via Featherless AI | https://huggingface.co                             |
| Qdrant      | Vector database                  | https://qdrant.tech (local Docker is fine for dev) |
| Mem0        | Persistent memory service        | https://mem0.ai                                    |
| Firecrawl   | Web content acquisition          | https://firecrawl.dev                              |

---

## Local Setup

Follow these steps to get a fully working development environment.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/deep-research-agent.git
cd deep-research-agent
```

### 2. Install Python Dependencies

```bash
uv sync
```

This command:

- Reads `pyproject.toml` and `uv.lock`
- Installs all dependencies (including dev tools: pytest, black, ruff, mypy)
- Creates a virtual environment in `.venv`

### 3. Start Qdrant (Vector Database)

The project includes a Docker Compose configuration. Start Qdrant with:

```bash
docker compose up qdrant -d
```

This starts a self-hosted Qdrant instance on `http://localhost:6333` (no authentication required for local development).

To verify it's running:

```bash
curl http://localhost:6333/health
```

You should see a JSON response with `"status": "ok"`.

### 4. Configure Environment Variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your real API keys. See [Environment Variables](#environment-variables) for the full reference table and which keys are required vs. optional.

### 5. Run the API Server (Optional)

To test the FastAPI app locally:

```bash
uv run fastapi dev src/research_agent/api/main.py
```

The API will be available at `http://localhost:8000`. OpenAPI documentation is at `http://localhost:8000/docs`.

### 6. Verify Everything Works

Run the full test suite to confirm your setup:

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95
```

All tests should pass with 95%+ coverage.

---

## Environment Variables

All configuration is loaded from environment variables (or a `.env` file) at application startup. The `pydantic-settings` library handles this in `src/research_agent/config.py`.

### Variable Reference

| Variable                 | Required | Default                  | Description                                                                                                                                                         |
| ------------------------ | -------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HF_TOKEN`               | **Yes**  | —                        | HuggingFace API token; used to authenticate with Featherless AI for LLM inference. Generate at https://huggingface.co/settings/tokens. Typically starts with `hf_`. |
| `QDRANT_URL`             | No       | `http://localhost:6333`  | URL of the Qdrant instance. For local development, the Docker instance listens on this URL. For cloud deployments, use your cluster URL.                            |
| `QDRANT_API_KEY`         | No       | `None`                   | Qdrant API key. Required only for cloud deployments; omit for local unauthenticated instances.                                                                      |
| `QDRANT_COLLECTION`      | No       | `research_chunks`        | Name of the Qdrant collection storing research embeddings.                                                                                                          |
| `QDRANT_VECTOR_SIZE`     | No       | `1024`                   | Embedding vector dimension. Must match the output size of `EMBEDDING_MODEL` (typically 1024 for BAAI/bge-large-en-v1.5).                                            |
| `EMBEDDING_MODEL`        | No       | `BAAI/bge-large-en-v1.5` | HuggingFace model ID for dense embeddings. Must produce vectors of size `QDRANT_VECTOR_SIZE`.                                                                       |
| `LLM_MODEL`              | No       | `Qwen/Qwen3-32B`         | Model served via Featherless AI for agent reasoning.                                                                                                                |
| `LLM_MAX_TOKENS`         | No       | `4096`                   | Maximum tokens to generate per LLM completion.                                                                                                                      |
| `MEM0_API_KEY`           | **Yes**  | —                        | API key for Mem0 persistent memory service. Obtain from https://mem0.ai.                                                                                            |
| `MEM0_RETENTION_DAYS`    | No       | `30`                     | Days before Mem0 session data expires.                                                                                                                              |
| `FIRECRAWL_API_KEY`      | **Yes**  | —                        | API key for Firecrawl web content acquisition. Obtain from https://firecrawl.dev.                                                                                   |
| `SECRET_KEY`             | **Yes**  | —                        | Secret key for signing JWT tokens (FastAPI auth). Must be at least 32 bytes. Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`              |
| `JWT_ALGORITHM`          | No       | `HS256`                  | JWT signing algorithm.                                                                                                                                              |
| `JWT_EXPIRY_MINUTES`     | No       | `60`                     | JWT token expiry in minutes.                                                                                                                                        |
| `RETRIEVAL_TOP_K`        | No       | `20`                     | Number of candidates to retrieve from Qdrant before reranking.                                                                                                      |
| `RETRIEVAL_RERANK_TOP_N` | No       | `5`                      | Number of results to return after FlashRank reranking.                                                                                                              |
| `LOG_LEVEL`              | No       | `INFO`                   | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.                                                                                              |
| `LOG_JSON`               | No       | `false`                  | If `true`, emit structured JSON logs (recommended in production).                                                                                                   |
| `ENVIRONMENT`            | No       | `dev`                    | Deployment environment: `dev`, `staging`, or `prod`. Controls logging and error detail.                                                                             |
| `APP_VERSION`            | No       | `0.1.0`                  | Application version, exposed in health check responses.                                                                                                             |
| `LANGCHAIN_TRACING_V2`   | No       | `false`                  | Enable LangSmith tracing for LangGraph nodes and LLM calls.                                                                                                         |
| `LANGCHAIN_API_KEY`      | No       | `None`                   | LangSmith API key. Required if `LANGCHAIN_TRACING_V2=true`.                                                                                                         |

### Example `.env` File

```bash
# Required secrets
HF_TOKEN=hf_abc1234567890...
MEM0_API_KEY=mem0_abc1234567890...
FIRECRAWL_API_KEY=firecrawl_abc1234567890...
SECRET_KEY=a1b2c3d4e5f6...64hexchars...

# Vector store (local Qdrant)
QDRANT_URL=http://localhost:6333

# LLM & retrieval config
LLM_MODEL=Qwen/Qwen3-32B
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RETRIEVAL_TOP_K=20
RETRIEVAL_RERANK_TOP_N=5

# Observability
LOG_LEVEL=DEBUG
ENVIRONMENT=dev
```

### Validation

The application validates all secrets at startup and will refuse to start if:

- Any required secret is missing
- A secret contains a placeholder value like `changeme`, `your-key-here`, `todo`, etc.
- `SECRET_KEY` is less than 32 bytes

This fail-fast design prevents accidentally deploying with incomplete credentials.

---

## Running the Test Suite

The project uses **pytest** as the test framework with strict coverage requirements (95% minimum).

### Test Organization

Tests are organized into two categories via pytest markers:

| Marker                     | Scope                                       | Speed | I/O           | When to Use                        |
| -------------------------- | ------------------------------------------- | ----- | ------------- | ---------------------------------- |
| `@pytest.mark.unit`        | Individual functions, utilities, components | Fast  | Mocked        | Most tests; run frequently         |
| `@pytest.mark.integration` | API endpoints, database operations          | Slow  | Real services | Critical flows; run before commits |

### Running Tests

#### All Tests

```bash
uv run pytest
```

This runs all unit and integration tests with coverage reporting.

#### Unit Tests Only

```bash
uv run pytest -m unit
```

Fast, isolated tests — safe to run continuously. Everything is mocked at the boundary (no real I/O).

#### Integration Tests Only

```bash
uv run pytest -m integration
```

Tests that hit real services (Qdrant, Mem0, etc.). Slower, requires full environment setup. Recommended before committing.

#### Run with Coverage Report

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95
```

Output shows:

- Line-by-line coverage (which lines are untested)
- Summary stats
- **Fails if coverage drops below 95%**

#### Run with Verbose Output

```bash
uv run pytest -v
```

Shows each test name as it runs. Helpful for debugging.

### Test Fixtures

Shared fixtures are defined in `tests/conftest.py`:

- `isolate_settings_env` (autouse) — Removes all known environment variables before each test, ensuring test isolation
- `clear_settings_cache` (autouse) — Clears the `get_settings()` cache before and after each test

These ensure that tests do not leak state and that patched environment variables are picked up correctly.

### Coverage Requirements

- **Minimum: 95%** — CI will fail if coverage drops below this
- No `# pragma: no cover` without explicit comment explaining why and approval in code review
- Cover happy paths, error paths, and edge cases

---

## Linting and Formatting

The project uses three non-negotiable tools to enforce code quality:

### black (Code Formatter)

Enforces consistent code formatting — no debate, no configuration.

```bash
uv run black src tests
```

Configuration (in `pyproject.toml`):

- Line length: 100 characters
- Target Python: 3.12

### ruff (Linter + Import Sorter)

Catches bugs, code smells, and performance issues.

```bash
uv run ruff check src tests --fix
```

Configuration (in `pyproject.toml`):

- Selects: E (errors), W (warnings), F (pyflakes), I (isort), UP (pyupgrade), B (bugbear), SIM (simplify)
- Ignores: E501 (line length — handled by black)
- Import sorting: `known-first-party = ["research_agent"]`

### mypy (Static Type Checker)

Enforces strict type safety — catches `None` type errors, missing annotations, etc.

```bash
uv run mypy src
```

Configuration (in `pyproject.toml`):

- **strict mode**: All functions must be fully annotated
- Plugins: `pydantic.mypy` for Pydantic model support
- Excludes tests (type checking tests is often overkill)
- Ignores missing type stubs for `flashrank`, `rank_bm25`, `mem0` (third-party packages without stubs)

### What Each Tool Catches

| Issue                                 | Tool      | Fix                          |
| ------------------------------------- | --------- | ---------------------------- |
| Inconsistent indentation, line length | **black** | Auto-fix: `black src tests`  |
| Unused imports, undefined names       | **ruff**  | Auto-fix: `ruff check --fix` |
| Type mismatches, missing annotations  | **mypy**  | Manual: review and fix       |
| Import order                          | **ruff**  | Auto-fix: `ruff check --fix` |

---

## Full Quality Gate

Before committing or opening a PR, run the complete quality check:

```bash
uv run black src tests && \
  uv run ruff check src tests --fix && \
  uv run mypy src && \
  uv run pytest --cov=src --cov-fail-under=95
```

Or as a single command in your shell:

```bash
uv run black src tests && uv run ruff check src tests --fix && uv run mypy src && uv run pytest --cov=src --cov-fail-under=95
```

**All four must pass before merge.** The CI/CD pipelines enforce this automatically.

---

## CI/CD Pipelines

The project uses GitHub Actions for continuous integration. All workflows are in `.github/workflows/`.

### 1. Tests Workflow (`.github/workflows/test.yml`)

**Trigger:** On push to `main` and on pull requests to `main`

**What it does:**

1. Checks out code
2. Installs Python 3.12 and uv
3. Installs all dependencies (including dev)
4. Runs: `uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95 --junitxml=test-results.xml`
5. Uploads test results as an artifact (visible in PR checks)

**Failure:** PR cannot merge if tests fail or coverage drops below 95%.

### 2. Lint Workflow (`.github/workflows/lint.yml`)

**Trigger:** On push to `main` and on pull requests to `main`

**What it does:**

1. Checks out code
2. Installs Python 3.12 and uv
3. Installs all dependencies
4. Runs: `uv run black --check src tests`
5. Runs: `uv run ruff check src tests`
6. Runs: `uv run mypy src`

**Failure:** PR cannot merge if any linter fails. (Fix with `black` and `ruff --fix` locally, then push again.)

### 3. Build Workflow (`.github/workflows/build.yml`)

**Trigger:** On push to `main` only (not on PRs)

**What it does:**

1. Checks out code
2. Logs in to GitHub Container Registry
3. Extracts Docker metadata (latest tag for main, SHA-based tags for commits)
4. Builds and pushes Docker image to `ghcr.io/<owner>/<repo>`

**Use:** For Railway deployments. The image includes all dependencies and the application code.

**Image tags:**

- `sha-<commit-hash>` — Immutable reference to this exact commit
- `latest` — Points to the most recent main branch build

### 4. Security Workflow (`.github/workflows/security.yml`)

**Trigger:** On push to `main`, on pull requests to `main`, and **weekly on Monday at 08:00 UTC**

**What it does:**

1. Checks out code
2. Installs dependencies
3. Runs **bandit** (OWASP security patterns): `uv run bandit -r src --severity-level=medium -f json`
4. Runs **pip-audit** (CVE dependency scan): `uv run pip-audit --format=json`
5. Uploads results as JSON artifacts (visible in PR checks)

**Configuration:** `continue-on-error: true` — Does not block merges, but results are visible and should be reviewed.

---

## Test-Driven Development Workflow

This project enforces **TDD (Test-Driven Development)** as a non-negotiable practice.

### The RED → GREEN → REFACTOR Cycle

#### 1. RED — Write a Failing Test

Write a test first that describes the desired behavior. It should fail because the feature doesn't exist yet.

```python
# tests/unit/test_my_feature.py
@pytest.mark.unit
def test_my_new_function_returns_expected_result():
    from research_agent.my_module import my_new_function
    result = my_new_function("input")
    assert result == "expected output"
```

Run the test — it **fails** (RED):

```bash
uv run pytest tests/unit/test_my_feature.py -v
```

#### 2. GREEN — Write Minimal Implementation

Write the **minimum code** to make the test pass. No extra features, no optimization.

```python
# src/research_agent/my_module.py
def my_new_function(x: str) -> str:
    return "expected output"
```

Run the test — it **passes** (GREEN):

```bash
uv run pytest tests/unit/test_my_feature.py -v
```

#### 3. REFACTOR — Improve (Keeping Tests Green)

Now that tests pass, improve the code:

- Remove duplication
- Extract utilities
- Improve readability
- Make it production-ready

**Keep running tests after each change** to ensure nothing breaks.

#### 4. Verify Coverage

Always check that your new code is covered:

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95
```

If coverage drops below 95%, add more tests for uncovered lines.

### Coverage Requirements

- **Never drop below 95%** — CI enforces this
- Cover happy paths (tests that verify correct behavior)
- Cover error paths (tests that verify exceptions are raised correctly)
- Cover edge cases (boundary conditions, empty inputs, etc.)

### Example: Adding a New Retriever

Suppose you want to add a new retriever implementation.

**1. Write tests first (RED):**

```python
# tests/unit/test_elasticsearch_retriever.py
@pytest.mark.unit
class TestElasticsearchRetriever:
    @pytest.fixture
    def retriever(self, mocker):
        mock_client = mocker.MagicMock()
        return ElasticsearchRetriever(mock_client)

    async def test_retrieve_returns_search_results(self, retriever):
        results = await retriever.retrieve("query", top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_retrieve_raises_on_empty_query(self, retriever):
        with pytest.raises(ValueError):
            await retriever.retrieve("", top_k=5)
```

Run tests (they fail):

```bash
uv run pytest tests/unit/test_elasticsearch_retriever.py -v
```

**2. Implement the retriever (GREEN):**

```python
# src/research_agent/retrieval/elasticsearch.py
from research_agent.retrieval.protocols import Retriever, SearchResult

class ElasticsearchRetriever:
    def __init__(self, client) -> None:
        self._client = client

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        if not query:
            raise ValueError("Query cannot be empty")
        # Call Elasticsearch...
        return [SearchResult(...) for _ in range(top_k)]
```

Run tests (they pass):

```bash
uv run pytest tests/unit/test_elasticsearch_retriever.py -v
```

**3. Refactor and add integration tests:**

```python
# tests/integration/test_elasticsearch_retriever.py
@pytest.mark.integration
class TestElasticsearchRetrieverIntegration:
    async def test_retrieve_from_real_elasticsearch_instance(self):
        # Uses a real or test Elasticsearch instance
        client = AsyncElasticsearch(hosts=["localhost:9200"])
        retriever = ElasticsearchRetriever(client)
        results = await retriever.retrieve("machine learning", top_k=10)
        assert len(results) > 0
        await client.close()
```

**4. Verify coverage:**

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95
```

---

## Adding New Components

The project follows a protocol-driven architecture. Adding new functionality means implementing a protocol and registering it with the application.

### Adding a New Tool

Tools are plugins that the agent can invoke to gather information or perform actions (e.g., Firecrawl for web scraping).

#### Checklist

1. **Create the tool file**

   ```
   src/research_agent/tools/<tool_name>.py
   ```

2. **Implement the `Tool` protocol**

   ```python
   from research_agent.tools.protocols import Tool, ToolResult

   class MyNewTool:
       name: str = "my_tool"
       description: str = "Does something useful"

       async def execute(self, input: dict) -> ToolResult:
           # Implement the tool logic
           return ToolResult(content="result", success=True)
   ```

3. **Write unit tests first (TDD)**

   ```
   tests/unit/test_my_new_tool.py
   ```

   Mock the external service at the boundary. Example:

   ```python
   @pytest.mark.unit
   async def test_my_new_tool_handles_api_error(mocker):
       mock_client = mocker.AsyncMock()
       mock_client.some_method.side_effect = Exception("API error")
       tool = MyNewTool(mock_client)
       result = await tool.execute({"param": "value"})
       assert result.success is False
   ```

4. **Register in the agent**
   - Edit `src/research_agent/agent/graph.py`
   - Add the tool to the agent's tool list during graph construction

5. **Add environment variables**
   - Add any required API keys to `src/research_agent/config.py`
   - Add defaults and descriptions to the `Settings` class
   - Document in `.env.example`

6. **Document security implications**
   - Edit `SECURITY.md` under "External Integrations"
   - List API keys, rate limits, and data handling

#### Example: Adding Firecrawl-like Tool

```python
# src/research_agent/tools/my_crawler.py
from research_agent.tools.protocols import Tool, ToolResult
from research_agent.config import get_settings

@dataclass
class CrawlInput:
    url: str
    max_pages: int = 10

class MyCrawlerTool:
    name: str = "my_crawler"
    description: str = "Crawl a website and extract content"

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.my_crawler_api_key.get_secret_value()

    async def execute(self, input: dict) -> ToolResult:
        crawl_input = CrawlInput(**input)
        try:
            content = await self._crawl(crawl_input.url, crawl_input.max_pages)
            return ToolResult(content=content, success=True)
        except Exception as e:
            return ToolResult(content=str(e), success=False)

    async def _crawl(self, url: str, max_pages: int) -> str:
        # Implementation...
        pass
```

### Adding a New Retriever

Retrievers fetch and rank documents from the knowledge base (Qdrant + BM25 hybrid search, etc.).

#### Checklist

1. **Create the retriever file**

   ```
   src/research_agent/retrieval/<retriever_name>.py
   ```

2. **Implement the `Retriever` protocol**

   ```python
   from research_agent.retrieval.protocols import Retriever, SearchResult

   class MyRetriever:
       async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
           # Query the backend and return results
           pass
   ```

3. **Write unit tests (mock the backend)**

   ```
   tests/unit/test_my_retriever.py
   ```

4. **Write integration tests (real backend)**

   ```
   tests/integration/test_my_retriever.py
   ```

5. **Make it swappable via config**
   - In `src/research_agent/agent/graph.py`, accept the retriever as a dependency injection parameter
   - Load it based on a config value if needed

#### Example: Elasticsearch Retriever

```python
# src/research_agent/retrieval/elasticsearch.py
from elasticsearch import AsyncElasticsearch
from research_agent.retrieval.protocols import Retriever, SearchResult

class ElasticsearchRetriever:
    def __init__(self, client: AsyncElasticsearch, index: str) -> None:
        self._client = client
        self._index = index

    async def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        response = await self._client.search(
            index=self._index,
            body={
                "query": {"multi_match": {"query": query}},
                "size": top_k,
            }
        )
        results = []
        for hit in response["hits"]["hits"]:
            results.append(SearchResult(
                content=hit["_source"]["content"],
                url=hit["_source"].get("url", ""),
                score=hit["_score"],
                metadata=hit["_source"].get("metadata", {}),
            ))
        return results
```

### Adding a New LLM Backend

The LLM backend handles model inference. Currently, the project uses HuggingFace Hub + Featherless AI.

#### Checklist

1. **Implement the `LLMClient` protocol**

   ```
   src/research_agent/llm/<backend_name>.py
   ```

   ```python
   from research_agent.llm.protocols import LLMClient

   class MyLLMClient:
       async def complete(self, messages: list[Message]) -> str:
           # Call the LLM API
           pass
   ```

2. **Write unit tests (mock the API)**

   ```
   tests/unit/test_my_llm_client.py
   ```

3. **Add environment variables**
   - Example: `MY_LLM_API_KEY`, `MY_LLM_MODEL`
   - Add to `src/research_agent/config.py` and `.env.example`

4. **Register in the agent**
   - Edit `src/research_agent/agent/graph.py`
   - Instantiate the client and pass it to the nodes that need it

#### Example: OpenAI Backend

```python
# src/research_agent/llm/openai.py
from openai import AsyncOpenAI
from research_agent.llm.protocols import LLMClient
from research_agent.config import get_settings

class OpenAIClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.llm_model

    async def complete(self, messages: list[dict]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
        )
        return response.choices[0].message.content
```

---

## Project Layout

The project is organized by feature/subsystem, not by layer (e.g., models, utils, etc.).

```
deep-research-agent/
├── CLAUDE.md                          # Architecture decisions, core principles
├── SECURITY.md                        # Security guidelines, CVE scanning
├── pyproject.toml                     # Dependencies, tool config (uv, black, ruff, mypy, pytest)
├── uv.lock                            # Lockfile; commit to repo (deterministic installs)
├── .env.example                       # Template with all required env vars
├── .python-version                    # Python 3.12
├── Dockerfile                         # Railway deployment
├── docker-compose.yml                 # Local Qdrant instance
│
├── docs/
│   └── development.md                 # ← You are here
│
├── src/research_agent/
│   ├── __init__.py
│   ├── py.typed                       # Marker: this is a typed package
│   ├── config.py                      # Settings via pydantic-settings
│   ├── logging.py                     # Structured logging setup
│   │
│   ├── agent/                         # LangGraph ReAct loop orchestration
│   │   ├── __init__.py
│   │   ├── state.py                   # AgentState dataclass
│   │   ├── graph.py                   # Graph construction, node registration
│   │   └── nodes.py                   # Individual node implementations
│   │
│   ├── retrieval/                     # Hybrid vector+BM25 search
│   │   ├── __init__.py
│   │   ├── protocols.py               # Retriever, Reranker protocols + SearchResult dataclass
│   │   ├── hybrid.py                  # HybridRetriever (Qdrant + BM25 + RRF)
│   │   └── reranker.py                # FlashRankReranker
│   │
│   ├── llm/                           # LLM inference abstraction
│   │   ├── __init__.py
│   │   ├── protocols.py               # LLMClient protocol
│   │   └── huggingface.py             # HuggingFaceClient (Featherless AI)
│   │
│   ├── memory/                        # Persistent memory across sessions
│   │   ├── __init__.py
│   │   ├── protocols.py               # MemoryService protocol
│   │   └── mem0.py                    # Mem0MemoryService implementation
│   │
│   ├── tools/                         # Agent tools (plugins)
│   │   ├── __init__.py
│   │   ├── protocols.py               # Tool protocol + ToolResult dataclass
│   │   └── firecrawl.py               # FirecrawlTool (via MCP)
│   │
│   ├── models/                        # Shared data models (no business logic)
│   │   ├── __init__.py
│   │   └── research.py                # ResearchQuery, ResearchReport, etc.
│   │
│   └── api/                           # FastAPI application
│       ├── __init__.py
│       ├── main.py                    # App factory: create_app()
│       ├── middleware.py              # Auth, request ID, rate limiting
│       └── routes/
│           ├── __init__.py
│           └── research.py            # POST /research endpoint
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures (isolate_settings_env, clear_settings_cache)
│   │
│   ├── unit/                          # Fast, mocked, no I/O
│   │   ├── __init__.py
│   │   ├── test_config.py             # Settings validation, defaults, caching
│   │   └── test_*.py                  # One file per module
│   │
│   └── integration/                   # Real services (Qdrant, Mem0)
│       ├── __init__.py
│       └── test_*.py                  # Full end-to-end flows
│
└── .github/workflows/                 # CI/CD
    ├── test.yml                       # Run tests + coverage
    ├── lint.yml                       # black, ruff, mypy
    ├── build.yml                      # Docker image build & push
    └── security.yml                   # bandit + pip-audit (weekly)
```

### Key Architecture Principles

1. **Protocols everywhere** — Every external dependency is hidden behind a protocol
   - `Retriever`, `Reranker`, `LLMClient`, `MemoryService`, `Tool`
   - Implementations are swappable without changing calling code

2. **Small, focused files** — Each file does one thing well
   - Most files are 200–400 lines
   - Max 800 lines (split into two files if larger)

3. **Async-first** — All I/O is async
   - Use `AsyncQdrantClient`, `AsyncInferenceClient`, etc.
   - No `requests` library (use `httpx.AsyncClient`)
   - LangGraph nodes are `async def`

4. **Type annotations mandatory** — Every function signature is fully annotated
   - No `Any` unless interfacing with untyped third-party libraries (with comments)

5. **Dataclasses and Pydantic models** — Never pass raw dicts internally
   - Use `@dataclass` for immutable value objects
   - Use Pydantic `BaseModel` for API request/response models

---

## Dependency Management

The project uses **uv** for deterministic, lockfile-based dependency management.

### Core Commands

#### Install All Dependencies (Including Dev)

```bash
uv sync
```

Creates a virtual environment in `.venv` and installs all dependencies from `uv.lock`.

#### Add a New Dependency

```bash
uv add package_name
```

This:

1. Adds `package_name` to `pyproject.toml` under `dependencies`
2. Updates `uv.lock` with the new package and all transitive dependencies
3. Installs it locally

#### Add a Dev-Only Dependency

```bash
uv add --dev package_name
```

Adds to `[project.optional-dependencies] dev` instead.

#### Update a Dependency

```bash
uv lock --upgrade package_name
```

Updates a specific package and its dependencies in `uv.lock`.

#### Sync After Pulling Changes

```bash
uv sync
```

Installs any new dependencies that teammates added (reads from `uv.lock`).

### Lockfile Policy

- **Commit `uv.lock` to Git** — It ensures everyone installs the exact same versions
- Never modify `uv.lock` manually
- Always use `uv add` and `uv lock` to update dependencies

### Current Dependencies

#### Core Runtime

| Package             | Version | Purpose                                              |
| ------------------- | ------- | ---------------------------------------------------- |
| `langgraph`         | ≥0.3    | Orchestration (ReAct loop, state machine)            |
| `huggingface-hub`   | ≥0.27   | LLM inference via Featherless AI (`InferenceClient`) |
| `qdrant-client`     | ≥1.12   | Vector database (async client)                       |
| `rank-bm25`         | ≥0.2    | BM25 sparse retrieval                                |
| `numpy`             | ≥1.26   | Numerical operations (embedding pooling)             |
| `flashrank`         | ≥0.2    | Result reranking                                     |
| `mem0ai`            | ≥0.1    | Persistent memory service                            |
| `mcp[cli]`          | ≥1.5    | Model Context Protocol (Firecrawl)                   |
| `fastapi[standard]` | ≥0.115  | REST API framework                                   |
| `pydantic-settings` | ≥2.7    | Environment variable loading                         |
| `httpx`             | ≥0.28   | Async HTTP client                                    |
| `pyjwt`             | ≥2.10   | JWT authentication                                   |
| `structlog`         | ≥24.4.0 | Structured logging                                   |

#### Dev Tools

| Package          | Version | Purpose                |
| ---------------- | ------- | ---------------------- |
| `pytest`         | ≥8.3    | Test runner            |
| `pytest-cov`     | ≥6.0    | Coverage reporting     |
| `pytest-asyncio` | ≥0.24   | Async test support     |
| `black`          | ≥25.1   | Code formatter         |
| `ruff`           | ≥0.9    | Linter + import sorter |
| `mypy`           | ≥1.15   | Type checker           |
| `types-pyjwt`    | —       | Type stubs for PyJWT   |
| `bandit[toml]`   | ≥1.8    | Security scanning      |
| `pip-audit`      | ≥2.6    | CVE dependency audit   |

### Version Strategy

The project uses **flexible pinning**:

- `package>=X.Y` — Accept any patch version ≥ X.Y
- Kept conservative to avoid breaking changes
- Updated quarterly or when security issues arise

---

## Troubleshooting

### Qdrant Connection Fails

**Error:** `ConnectionError: Failed to connect to http://localhost:6333`

**Fix:**

```bash
docker compose up qdrant -d
docker logs qdrant  # Check logs
curl http://localhost:6333/health  # Verify it's running
```

### Secret Validation Fails

**Error:** `ValueError: Required secret 'HF_TOKEN' is not set`

**Fix:**

- Ensure `.env` file exists in the project root
- All required secrets are set (see [Environment Variables](#environment-variables))
- No placeholder values like `changeme` or `todo`

### Tests Fail Randomly

**Error:** Tests pass locally but fail in CI, or fail inconsistently

**Cause:** Usually test isolation issue (state leaking between tests)

**Fix:**

- Check that fixtures clear caches: `get_settings.cache_clear()`
- Ensure mocks are reset between tests
- Run tests in random order: `uv run pytest --random-order`

### Coverage Below 95%

**Error:** `FAILED: Coverage ≥ 95% required`

**Fix:**

1. Find uncovered lines: `uv run pytest --cov=src --cov-report=term-missing`
2. Add tests for those lines
3. Re-run coverage check

### Type Checking Fails

**Error:** `error: Name 'X' is not defined` from mypy

**Fix:**

1. Ensure the variable/import is defined
2. Add type annotations to function signatures
3. If a third-party package lacks stubs, it should be in the `mypy.overrides` in `pyproject.toml`

---

## Next Steps

- Read [`CLAUDE.md`](../CLAUDE.md) for architecture decisions and core principles
- Read [`SECURITY.md`](../SECURITY.md) for security guidelines and CVE scanning
- Check out existing implementations in `src/research_agent/` to understand the pattern
- Write your first test (RED) before implementing any feature
- Run the full quality gate before committing

Happy developing!
