# Deep Research Agent

A modular, extensible AI research agent that accepts natural-language queries and returns structured, cited reports using hybrid retrieval, agentic reasoning, and persistent memory.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000)](#development)
[![Type Checking](https://img.shields.io/badge/Type%20Checking-mypy-1e90ff)](#development)

---

## Overview

The Deep Research Agent is a production-ready system that orchestrates a ReAct (Reasoning + Acting) loop to conduct deep, multi-turn research on arbitrary topics. It combines:

- **Hybrid Retrieval**: Dense embeddings + BM25 sparse search via Qdrant, fused with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: FlashRank for improved relevance ordering
- **Agentic Reasoning**: Qwen3-32B via Featherless AI for complex reasoning and tool selection
- **Persistent Memory**: Mem0 for cross-session context accumulation
- **Web Acquisition**: Firecrawl via MCP for live web search and content scraping
- **Async-First Design**: Handles multiple concurrent research sessions without blocking

### Key Features

- **Structured Reports**: JSON responses with synthesized summaries and citation trails
- **Session Continuity**: Memory persists across requests within a session for context-aware follow-ups
- **FastAPI Service**: RESTful API with JWT authentication, rate limiting, and request tracking
- **Production Ready**: Designed for Railway deployment with health checks and observability
- **Protocol-Based Abstraction**: Every external dependency (LLM, storage, memory, tools) is swappable
- **95% Test Coverage**: Comprehensive unit and integration test suite

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker** (for running Qdrant locally)
- **uv** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Installation

```bash
# Clone the repository
git clone https://github.com/objones25/deep-research-agent.git
cd deep-research-agent

# Install dependencies
uv sync

# Start Qdrant vector database
docker compose up qdrant -d

# Verify Qdrant is running
curl http://localhost:6333/health

# Copy environment template and fill in API keys
cp .env.example .env
# Edit .env with your credentials for HF_TOKEN, MEM0_API_KEY, FIRECRAWL_API_KEY, SECRET_KEY
```

### Running Locally

```bash
# Start the FastAPI development server
uv run fastapi dev src/research_agent/api/main.py

# In another terminal, test the API
curl -X POST http://localhost:8000/research \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest advances in retrieval-augmented generation?",
    "session_id": "session-123",
    "max_iterations": 5
  }'
```

### Generating a JWT Token

```bash
python3 << 'EOF'
import jwt
import time

secret = "your-secret-key-from-.env"
payload = {"sub": "user@example.com", "exp": int(time.time()) + 3600}
token = jwt.encode(payload, secret, algorithm="HS256")
print(f"Authorization: Bearer {token}")
EOF
```

---

## Architecture

### Component Overview

| Component        | Purpose                   | Technology                 |
| ---------------- | ------------------------- | -------------------------- |
| Orchestration    | ReAct loop state machine  | LangGraph                  |
| LLM Inference    | Reasoning and tool calls  | Qwen3-32B (Featherless AI) |
| Dense Embeddings | Semantic vector search    | BAAI/bge-large-en-v1.5     |
| Sparse Search    | Exact term matching       | BM25 via rank-bm25         |
| Vector DB        | Hybrid retrieval backend  | Qdrant (self-hosted)       |
| Reranking        | Cross-encoder re-scoring  | FlashRank                  |
| Memory           | Persistent session recall | Mem0 (managed service)     |
| Web Tools        | Live search + scraping    | Firecrawl via MCP          |
| HTTP API         | RESTful service layer     | FastAPI                    |
| Logging          | Structured observability  | structlog                  |

### Data Flow

```
POST /research
    |
    v
[Initialize AgentState]
    |
    v
[Memory Retrieval] --> Fetch prior context from session
    |
    v
[Hybrid Retrieval] --> Dense + Sparse search, RRF fusion, FlashRank reranking
    |
    v
[Reasoning Loop] --> LLM generates tool calls or final answer
    |
    +---> [Tool Execution] --> Firecrawl search/scrape (if needed)
    |        |
    |        v
    +--- [Continue Loop] or [Synthesize]
    |
    v
[Synthesis] --> Final report + citations, persist summary to memory
    |
    v
JSON ResearchReport response
```

### High-Level Stack

- **Async-first** asyncio runtime (no blocking I/O)
- **Type-safe** fully annotated code with mypy strict mode
- **Protocol-driven** every external dependency is swappable
- **SOLID principles** small, focused classes with single responsibilities
- **Test-driven** 95%+ code coverage, unit + integration tests

For detailed architecture decisions, see [docs/architecture.md](docs/architecture.md).

---

## API Usage

### POST /research

Execute a research query and receive a structured report.

**Authentication**: Bearer JWT with `sub` claim  
**Rate Limit**: 60 requests per 60 seconds per IP  
**Timeout**: Configurable via LangGraph checkpointing

#### Request

```bash
curl -X POST http://localhost:8000/research \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are quantum error correction codes?",
    "session_id": "research-alice-2024",
    "max_iterations": 5
  }'
```

#### Response (200 OK)

```json
{
  "query": "What are quantum error correction codes?",
  "summary": "Quantum error correction codes are techniques for protecting quantum information from decoherence and operational errors...",
  "citations": [
    {
      "url": "https://arxiv.org/abs/2402.12345",
      "content_snippet": "Surface codes represent the most promising approach to practical QECC...",
      "relevance_score": 0.94
    }
  ],
  "session_id": "research-alice-2024"
}
```

### GET /health

Check service and dependency health.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "checks": {
    "qdrant": "ok"
  },
  "timestamp": "2024-04-15T14:32:10Z"
}
```

For complete API documentation, see [docs/api-reference.md](docs/api-reference.md).

---

## Documentation

- **[Architecture Guide](docs/architecture.md)** — System design, component contracts, data flow, hybrid retrieval pipeline, and design rationale
- **[API Reference](docs/api-reference.md)** — Endpoints, authentication, rate limiting, request/response schemas, error codes
- **[Development Guide](docs/development.md)** — Local setup, testing, linting, CI/CD, and component extension
- **[Module Reference](docs/modules.md)** — Public API of `research_agent` package, function signatures, protocols
- **[CLAUDE.md](CLAUDE.md)** — AI assistant guide, development principles, conventions, and extensibility patterns

---

## Development

### Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=95

# Run unit tests only
uv run pytest -m unit

# Run integration tests (requires live Qdrant)
uv run pytest -m integration

# Watch mode (requires pytest-watch)
uv run ptw
```

### Quality Checks

```bash
# Format code
uv run black src tests

# Lint and auto-fix
uv run ruff check src tests --fix

# Type checking (strict mode)
uv run mypy src

# Security scanning
uv run bandit -r src
uv run pip-audit
```

### Full Quality Gate

```bash
# Run everything: format, lint, type check, security scan, and tests
uv run black src tests && \
uv run ruff check src tests && \
uv run mypy src && \
uv run bandit -r src && \
uv run pytest --cov=src --cov-fail-under=95
```

### Adding a New Tool

1. Create `src/research_agent/tools/<tool_name>.py`
2. Implement the `Tool` protocol (name, description, execute method)
3. Write unit tests first (mock the external service)
4. Register in agent graph's tool list (`src/research_agent/agent/graph.py`)
5. Add environment variables to `config.py` and `.env.example`
6. Document in `SECURITY.md`

### Adding a New Retriever

1. Create `src/research_agent/retrieval/<retriever_name>.py`
2. Implement the `Retriever` protocol
3. Write unit tests (mock Qdrant) and integration tests (real local Qdrant)
4. Swap in via config — no changes to agent code required

---

## Project Structure

```
.
├── README.md                      # This file
├── CLAUDE.md                      # AI assistant guide and conventions
├── SECURITY.md                    # Security guidelines and integrations
├── pyproject.toml                 # Dependencies and tool configuration
├── .env.example                   # Environment variable template
├── Dockerfile                     # Production container image
├── docker-compose.yml             # Local development services
├── docs/
│   ├── architecture.md            # System design and component contracts
│   ├── api-reference.md           # Full API documentation
│   ├── development.md             # Developer setup and workflows
│   └── modules.md                 # Module reference and signatures
├── src/research_agent/
│   ├── agent/                     # LangGraph orchestration
│   │   ├── graph.py               # ReAct loop definition
│   │   ├── nodes.py               # Node implementations
│   │   └── state.py               # AgentState dataclass
│   ├── retrieval/                 # Hybrid search and reranking
│   │   ├── protocols.py           # Retriever and Reranker contracts
│   │   ├── hybrid.py              # HybridRetriever (dense + sparse)
│   │   └── reranker.py            # FlashRankReranker
│   ├── memory/                    # Persistent session memory
│   │   ├── protocols.py           # MemoryService contract
│   │   └── mem0.py                # Mem0 implementation
│   ├── tools/                     # External tools via MCP
│   │   ├── protocols.py           # Tool protocol
│   │   └── firecrawl.py           # Firecrawl search and scrape
│   ├── llm/                       # LLM inference
│   │   ├── protocols.py           # LLMClient contract
│   │   └── huggingface.py         # HuggingFace / Featherless AI
│   ├── models/                    # Shared dataclasses and DTOs
│   │   └── research.py            # ResearchQuery, ResearchReport, etc.
│   ├── api/                       # FastAPI service
│   │   ├── main.py                # App factory and startup
│   │   ├── routes/
│   │   │   └── research.py        # POST /research endpoint
│   │   ├── middleware.py          # Auth, rate limiting, request ID
│   │   └── dependencies.py        # Dependency injection utilities
│   ├── config.py                  # Pydantic Settings from env
│   └── logging.py                 # Structured logging setup
└── tests/
    ├── conftest.py                # Fixtures and mock factories
    ├── unit/                      # Fast isolated unit tests
    └── integration/               # Tests requiring live services
```

---

## Environment Variables

All configuration is loaded from `.env` or system environment at startup. See `.env.example` for the complete reference.

### Required

| Variable            | Purpose                    | Example                 |
| ------------------- | -------------------------- | ----------------------- |
| `HF_TOKEN`          | HuggingFace/Featherless AI | `hf_xxxxxxxxxxxxxxx`    |
| `SECRET_KEY`        | JWT signing key (32+ char) | `super-secret-key-...`  |
| `QDRANT_URL`        | Vector database URL        | `http://localhost:6333` |
| `QDRANT_API_KEY`    | Qdrant cloud auth          | `api-key-xxx`           |
| `MEM0_API_KEY`      | Mem0 managed memory        | `mem0-key-xxx`          |
| `FIRECRAWL_API_KEY` | Web scraping tool          | `fc-key-xxx`            |

### Optional

| Variable               | Default | Purpose                                     |
| ---------------------- | ------- | ------------------------------------------- |
| `ENVIRONMENT`          | `dev`   | `dev`, `staging`, or `prod` (controls docs) |
| `LOG_LEVEL`            | `INFO`  | Logging verbosity                           |
| `LOG_JSON`             | `false` | JSON logs for production aggregation        |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing (for debugging)    |
| `LANGCHAIN_API_KEY`    | —       | LangSmith API key (if tracing enabled)      |

---

## Deployment

### Railway

The application is designed for **Railway** deployment:

```bash
# Set environment variables in Railway UI or via .env
# Build and deploy
railway up

# View logs
railway logs
```

Railway automatically detects and runs the `Dockerfile`. Health checks target `GET /health`.

### Docker

Build and run a container locally:

```bash
# Build image
docker build -t deep-research-agent:latest .

# Run with environment variables
docker run -e SECRET_KEY=xxx -e HF_TOKEN=xxx ... -p 8000:8000 deep-research-agent:latest
```

---

## Design Principles

### Async-First

Every I/O method is `async`; the event loop is never blocked. Multiple concurrent research sessions fit in a single Railway dyno.

### Protocol-Based Abstraction

External dependencies (LLM, vector store, memory, tools) are hidden behind protocols. Swap implementations without changing agent code.

### Incremental Complexity

Start simple; add advanced patterns (multi-agent supervision, GraphRAG, adaptive routing) only when evidence exists that they're needed.

### Fail Early

Validate inputs at boundaries; raise immediately on precondition violations. No silent fallbacks unless explicitly specified.

### 95% Test Coverage

Minimum 95% code coverage enforced by CI. TDD is mandatory for new features.

---

## Contributing

Contributions welcome! Please follow these guidelines:

1. **Write tests first** (TDD: RED → GREEN → REFACTOR)
2. **Run quality checks** before pushing (`black`, `ruff`, `mypy`, `pytest`)
3. **Document protocols** and design decisions in docstrings
4. **Keep code focused**: files under 800 lines, functions under 50 lines
5. **See [docs/development.md](docs/development.md)** for detailed workflows

---

## License

MIT License — See LICENSE file for details.

---

## Support

For bugs, questions, or feature requests:

1. Check the [API Reference](docs/api-reference.md) and [Architecture Guide](docs/architecture.md)
2. Review [docs/development.md](docs/development.md) for common setup issues
3. Enable debug logging (`LOG_LEVEL=DEBUG`) and include the `X-Request-ID` from your request
4. Search existing issues and create a new one with relevant logs

---

## Acknowledgments

Built with:

- [LangGraph](https://github.com/langchain-ai/langgraph) — orchestration
- [Qdrant](https://qdrant.tech/) — vector database
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [Featherless AI](https://www.featherless.ai/) — LLM inference
- [Firecrawl](https://firecrawl.dev/) — web acquisition
- [Mem0](https://mem0.ai/) — persistent memory
