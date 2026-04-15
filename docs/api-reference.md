# Deep Research Agent API Reference

## Overview

The Deep Research Agent API is a RESTful service that accepts research queries and returns structured, cited reports using hybrid retrieval and agentic reasoning.

### Base URL

```
https://api.deepresearch.app
```

(Use `http://localhost:8000` for local development)

### API Version

The API does not use explicit versioning in the URL. Breaking changes are tracked via the app version in `X-App-Version` response headers.

### Content Type

All requests and responses use `Content-Type: application/json`.

---

## Authentication

The API uses **JWT (JSON Web Token) Bearer authentication** for protected endpoints. Tokens are validated using the HS256 algorithm by default.

### Token Format

Requests must include an `Authorization` header with a Bearer token:

```
Authorization: Bearer <JWT_TOKEN>
```

The JWT payload must contain a `sub` claim (subject), which identifies the user:

```json
{
  "sub": "user-id-or-name",
  "exp": 1234567890
}
```

### Generating Tokens

Tokens are signed using the `SECRET_KEY` environment variable. During deployment, ensure:

1. `SECRET_KEY` is a strong, random string (32+ characters)
2. `JWT_ALGORITHM` (default: `HS256`) matches the signing algorithm used to create tokens
3. Tokens include an `exp` claim for expiration (recommended)

Example token generation (outside the API):

```python
import jwt
import time

secret = "your-secret-key-here"
payload = {
    "sub": "user123",
    "exp": int(time.time()) + 3600  # Expires in 1 hour
}
token = jwt.encode(payload, secret, algorithm="HS256")
print(f"Authorization: Bearer {token}")
```

### Authentication Errors

Requests missing, malformed, or expired tokens receive a `401 Unauthorized` response:

```json
{
  "detail": "Invalid or missing authentication credentials."
}
```

The response also includes the `WWW-Authenticate: Bearer` header.

### Endpoints Requiring Authentication

- `POST /research` — requires Bearer JWT

### Endpoints Without Authentication

- `GET /health` — public, no authentication required
- `GET /docs` — public, Swagger UI (disabled in production)
- `GET /redoc` — public, ReDoc (disabled in production)

---

## Rate Limiting

The API enforces **per-IP rate limiting** using a sliding-window algorithm. The default limit is **60 requests per 60 seconds** per unique client IP.

### Algorithm

Timestamps of requests from each client IP are stored in a deque. On each new request:

1. Timestamps older than the window are evicted from the left side of the deque
2. If the remaining count equals or exceeds `max_requests`, the request is rejected with `429`
3. The current timestamp is appended to the deque

### Rate Limit Configuration

Rate limits are set at application startup:

- `max_requests`: 60
- `window_seconds`: 60.0

To adjust, modify `_rate_limiter` in `src/research_agent/api/dependencies.py`.

### Rate Limit Headers

Responses do not include `X-RateLimit-*` headers at this time. Rate limit status is communicated solely through HTTP status codes.

### Rate Limit Errors

Requests exceeding the rate limit receive a `429 Too Many Requests` response:

```json
{
  "detail": "Rate limit exceeded: maximum 60 requests per 60s window."
}
```

### IP Detection

The client IP is extracted from `request.client.host`. In production (behind a proxy), ensure the `X-Forwarded-For` header is correctly set, or configure the proxy to use the `Forwarded` standard header.

---

## Request ID Tracking

Every request and response includes a unique **request ID** in the `X-Request-ID` header. This ID is used for tracing requests through logs and debugging.

### Providing a Request ID

To provide your own request ID, include it in the request:

```
X-Request-ID: my-custom-request-id-123
```

### Auto-Generated Request IDs

If the header is omitted, the API generates a UUID v4 automatically:

```
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
```

### Request ID in Responses

The API echoes the request ID back in all responses:

```
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
```

### Logging

The request ID is automatically bound to all structured logs emitted during the request lifecycle via `contextvars`, so you can correlate logs across the service.

---

## Endpoints

### POST /research

Accept a research query and execute the agent to return a structured report.

**Authentication:** Required (Bearer JWT)  
**Rate Limiting:** Enforced (60 requests per 60 seconds per IP)  
**Content-Type:** `application/json`

#### Request Headers

| Header          | Required | Example              | Description                             |
| --------------- | -------- | -------------------- | --------------------------------------- |
| `Authorization` | Yes      | `Bearer eyJ0eXAi...` | JWT Bearer token with `sub` claim       |
| `X-Request-ID`  | No       | `my-request-id-123`  | Custom request ID; generated if omitted |
| `Content-Type`  | Yes      | `application/json`   | Must be `application/json`              |

#### Request Body

| Field            | Type    | Required | Default             | Constraints          | Description                                                     |
| ---------------- | ------- | -------- | ------------------- | -------------------- | --------------------------------------------------------------- |
| `query`          | string  | Yes      | —                   | Non-empty after trim | The research question to investigate                            |
| `session_id`     | string  | No       | Auto-generated UUID | Non-empty            | Optional session identifier for cross-request memory continuity |
| `max_iterations` | integer | No       | 5                   | 1–20 (inclusive)     | Maximum ReAct loop iterations; caps reasoning cycles            |

#### Example Request

```bash
curl -X POST http://localhost:8000/research \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
  -H "X-Request-ID: research-2024-001" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in quantum computing?",
    "session_id": "session-user-123",
    "max_iterations": 5
  }'
```

#### Response Body

| Field        | Type              | Description                                        |
| ------------ | ----------------- | -------------------------------------------------- |
| `query`      | string            | The original research question                     |
| `summary`    | string            | Synthesized research summary produced by the agent |
| `citations`  | array of Citation | Source citations supporting the summary            |
| `session_id` | string            | The session identifier used for this request       |

**Citation Object:**

| Field             | Type   | Description                                        |
| ----------------- | ------ | -------------------------------------------------- |
| `url`             | string | Source URL                                         |
| `content_snippet` | string | Relevant excerpt from the source (up to 500 chars) |
| `relevance_score` | number | Reranker relevance score (0.0 to 1.0)              |

#### Example Response

```json
{
  "query": "What are the latest developments in quantum computing?",
  "summary": "Recent breakthroughs in quantum computing include Google's achievement of quantum supremacy using their Willow chip, which demonstrated error rates below the threshold for surface codes. IBM has advanced its quantum processors to 1,121 qubits, while IonQ announced fault-tolerant quantum computing capabilities. These developments suggest a transition from the NISQ (Noisy Intermediate-Scale Quantum) era toward practical, scalable quantum systems.",
  "citations": [
    {
      "url": "https://blog.google/technology/ai/willow-quantum-chip/",
      "content_snippet": "Google's Willow quantum chip demonstrates error correction and quantum supremacy capabilities...",
      "relevance_score": 0.95
    },
    {
      "url": "https://www.ibm.com/quantum/roadmap",
      "content_snippet": "IBM's quantum roadmap outlines progress toward 1121-qubit systems by 2024...",
      "relevance_score": 0.87
    }
  ],
  "session_id": "session-user-123"
}
```

#### Status Codes

| Code  | Meaning               | Example Response                                             |
| ----- | --------------------- | ------------------------------------------------------------ |
| `200` | Success               | Valid `ResearchReportResponse` with summary and citations    |
| `400` | Bad Request           | Malformed query (empty after trim, invalid `max_iterations`) |
| `401` | Unauthorized          | Missing, malformed, or expired Bearer JWT                    |
| `422` | Validation Error      | Invalid request body schema (e.g., wrong field type)         |
| `429` | Too Many Requests     | Rate limit exceeded for this IP                              |
| `503` | Service Unavailable   | Agent runner not initialized (startup not complete)          |
| `500` | Internal Server Error | Unhandled exception in agent or dependencies                 |

#### Error Response Examples

**400 Bad Request (empty query):**

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "query must not be blank",
      "type": "value_error"
    }
  ]
}
```

**401 Unauthorized (missing token):**

```json
{
  "detail": "Invalid or missing authentication credentials."
}
```

**422 Unprocessable Entity (invalid max_iterations):**

```json
{
  "detail": [
    {
      "loc": ["body", "max_iterations"],
      "msg": "Input should be less than or equal to 20",
      "type": "less_than_equal"
    }
  ]
}
```

**429 Too Many Requests:**

```json
{
  "detail": "Rate limit exceeded: maximum 60 requests per 60s window."
}
```

**503 Service Unavailable:**

```json
{
  "detail": "Research agent is not available."
}
```

---

### GET /health

Check the health status of the service and its dependencies.

**Authentication:** Not required (public endpoint)  
**Rate Limiting:** Not enforced  
**Content-Type:** `application/json`

#### Request Headers

No required headers. The `X-Request-ID` header is optional.

| Header         | Optional | Example            | Description                             |
| -------------- | -------- | ------------------ | --------------------------------------- |
| `X-Request-ID` | Yes      | `health-check-001` | Custom request ID; generated if omitted |

#### Example Request

```bash
curl -X GET http://localhost:8000/health \
  -H "X-Request-ID: health-check-001"
```

#### Response Body

| Field       | Type   | Description                                                    |
| ----------- | ------ | -------------------------------------------------------------- |
| `status`    | string | Overall health status: `"ok"`, `"degraded"`, or `"error"`      |
| `checks`    | object | Per-dependency health check results (e.g., `{"qdrant": "ok"}`) |
| `timestamp` | string | ISO 8601 UTC timestamp of the health check                     |

**Dependency Checks:**

Currently, the service probes:

- `qdrant` — Qdrant vector database (via `GET {qdrant_url}/healthz` with 2-second timeout)

Future checks may include Mem0 memory service and other dependencies.

#### Example Response

```json
{
  "status": "ok",
  "checks": {
    "qdrant": "ok"
  },
  "timestamp": "2024-04-15T14:32:10.123456+00:00"
}
```

**With degraded dependency:**

```json
{
  "status": "error",
  "checks": {
    "qdrant": "error"
  },
  "timestamp": "2024-04-15T14:32:10.123456+00:00"
}
```

#### Status Codes

| Code  | Meaning   | Condition                                      |
| ----- | --------- | ---------------------------------------------- |
| `200` | Healthy   | All dependency checks return `"ok"`            |
| `503` | Unhealthy | One or more dependency checks return `"error"` |

---

## Error Responses

All error responses follow a consistent envelope format:

```json
{
  "detail": "Human-readable error message or validation details."
}
```

For validation errors (400/422), the `detail` field is an array of validation objects:

```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "error description",
      "type": "error_code"
    }
  ]
}
```

### Error Codes and Meanings

| HTTP Code | Error Case                     | Description                                                                | Recovery                                                    |
| --------- | ------------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `400`     | Invalid request body           | Query is blank, `max_iterations` out of range, or other validation failure | Correct the request and retry                               |
| `401`     | Missing/invalid authentication | Bearer token is missing, malformed, expired, or signature invalid          | Provide a valid JWT Bearer token                            |
| `422`     | Malformed JSON                 | Request body is not valid JSON or field types are wrong                    | Ensure JSON is well-formed and field types match the schema |
| `429`     | Rate limit exceeded            | Client IP has exceeded 60 requests per 60 seconds                          | Wait and retry after the sliding window passes              |
| `503`     | Service unavailable            | Agent runner not initialized (startup still in progress)                   | Retry after a few seconds; check service logs               |
| `500`     | Unhandled exception            | An unexpected error occurred in the agent or a dependency                  | Provide the `X-Request-ID` to support; check server logs    |

---

## Request/Response Examples

### Complete End-to-End Example

**1. Generate a JWT token:**

```bash
python3 << 'EOF'
import jwt
import time

secret = "my-super-secret-key-change-in-production"
payload = {
    "sub": "researcher@example.com",
    "exp": int(time.time()) + 3600
}
token = jwt.encode(payload, secret, algorithm="HS256")
print(token)
EOF
```

**2. Make a research request:**

```bash
curl -X POST http://localhost:8000/research \
  -H "Authorization: Bearer <token_from_above>" \
  -H "X-Request-ID: req-001" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of async programming?",
    "session_id": "my-session-001",
    "max_iterations": 3
  }'
```

**3. Expected response (200 OK):**

```json
{
  "query": "What are the benefits of async programming?",
  "summary": "Async programming enables non-blocking I/O and improved resource utilization. Benefits include higher concurrency (handling thousands of connections with few threads), better responsiveness (the application remains interactive during I/O waits), and efficient resource usage (threads are not blocked). Async is essential for high-throughput web services, real-time applications, and systems with frequent I/O operations like web servers and database clients.",
  "citations": [
    {
      "url": "https://example.com/async-guide",
      "content_snippet": "Async programming allows a single thread to handle thousands of concurrent operations...",
      "relevance_score": 0.92
    }
  ],
  "session_id": "my-session-001"
}
```

---

## App Factory Pattern

The API uses a **factory function pattern** to construct the FastAPI application. This design decision enables:

1. **Testability:** Tests can construct an app with a mock `AgentRunner` to avoid live agent invocations
2. **Dependency Injection:** The runner is injected at construction time, not lazily created
3. **Configuration Flexibility:** Different environments can pass different runners or configuration objects

### Implementation

```python
# src/research_agent/api/main.py

def create_app(agent_runner: AgentRunner | None = None) -> FastAPI:
    """Construct and configure a FastAPI application.

    Args:
        agent_runner: Optional runner injected for testing; if None,
            the app will return 503 on /research until a runner is set.

    Returns:
        A fully configured FastAPI instance.
    """
    app = FastAPI(...)
    app.state.agent_runner = agent_runner
    # ... wire up middleware, routes, etc.
    return app

# Module-level instance used by uvicorn
app = create_app()
```

### Production Deployment

In production, `uvicorn` invokes the module-level `app` instance. The runner is wired up during the **lifespan startup** phase (async context manager) or passed directly if needed.

### Testing

Tests call `create_app()` directly with a mock runner:

```python
@pytest.fixture
def app():
    mock_runner = AsyncMock(spec=AgentRunner)
    return create_app(agent_runner=mock_runner)

def test_research_endpoint(app):
    client = TestClient(app)
    response = client.post("/research", json={"query": "test"})
    assert response.status_code == 200
```

---

## OpenAPI Documentation

The API auto-generates OpenAPI (Swagger) documentation at two endpoints:

- **Swagger UI:** `/docs` (interactive API explorer)
- **ReDoc:** `/redoc` (read-only documentation)

Both are **disabled in production** (`environment == "prod"`). In development and staging, they are available without authentication.

### Accessing OpenAPI Docs

```
http://localhost:8000/docs
http://localhost:8000/redoc
```

The OpenAPI schema is available at:

```
http://localhost:8000/openapi.json
```

---

## Environment Variables

The API reads configuration from environment variables. All values are validated at startup via `pydantic-settings`.

### Required Variables

| Variable            | Type   | Default | Example                         | Description                            |
| ------------------- | ------ | ------- | ------------------------------- | -------------------------------------- |
| `SECRET_KEY`        | string | None    | `super-secret-key-min-32-chars` | JWT signing secret (must be 32+ chars) |
| `QDRANT_URL`        | string | None    | `http://qdrant:6333`            | Qdrant vector database URL             |
| `QDRANT_API_KEY`    | string | None    | `abc123def456`                  | Qdrant API key (if using cloud)        |
| `HF_TOKEN`          | string | None    | `hf_xxxxxxxxxxxx`               | HuggingFace token for Featherless AI   |
| `MEM0_API_KEY`      | string | None    | `mem0-key-xxx`                  | Mem0 API key                           |
| `FIRECRAWL_API_KEY` | string | None    | `fc-key-xxx`                    | Firecrawl API key                      |

### Optional Variables

| Variable               | Type   | Default   | Example        | Description                                              |
| ---------------------- | ------ | --------- | -------------- | -------------------------------------------------------- |
| `APP_VERSION`          | string | `"1.0.0"` | `"2.1.0"`      | Application version string                               |
| `LOG_LEVEL`            | string | `"INFO"`  | `"DEBUG"`      | Logging level                                            |
| `LOG_JSON`             | bool   | `false`   | `true`         | Output logs as JSON (structured logging)                 |
| `ENVIRONMENT`          | string | `"dev"`   | `"prod"`       | Deployment environment; controls OpenAPI docs visibility |
| `JWT_ALGORITHM`        | string | `"HS256"` | `"HS256"`      | JWT signing algorithm                                    |
| `LANGCHAIN_TRACING_V2` | string | None      | `"true"`       | Enable LangChain tracing                                 |
| `LANGCHAIN_API_KEY`    | string | None      | `"lc-key-xxx"` | LangChain API key (if tracing enabled)                   |

### Loading Variables

Variables are loaded from the environment in this order:

1. `.env` file (if present in the working directory)
2. System environment variables
3. Defaults (if defined in `Settings` dataclass)

Use a `.env` file for local development:

```bash
# .env
SECRET_KEY=my-super-secret-key-for-development
QDRANT_URL=http://localhost:6333
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
MEM0_API_KEY=mem0-key-xxx
FIRECRAWL_API_KEY=fc-key-xxx
ENVIRONMENT=dev
LOG_LEVEL=INFO
```

---

## Middleware Stack

The application includes the following middleware (applied in order):

### RequestIDMiddleware

- **Purpose:** Attach a unique ID to every request/response
- **Behavior:** Extracts or generates `X-Request-ID` header, binds it to request context for logging, and echoes it back in responses
- **Configuration:** None required; enabled by default

---

## Deployment Notes

### Container Image

The API is deployed as a Docker container. Refer to `Dockerfile` for the image specification.

### Railway Deployment

The application is designed for **Railway** deployment:

1. Environment variables are set via Railway UI or `.env` file
2. The startup command is `uvicorn src.research_agent.api.main:app --host 0.0.0.0 --port 8080`
3. Health checks target `GET /health`

### Lifespan Management

The application uses FastAPI's **lifespan context manager** to:

- Initialize structured logging on startup
- Activate LangChain tracing if configured
- No teardown required at this stage

---

## Glossary

| Term               | Definition                                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Bearer Token**   | A token sent in the `Authorization: Bearer <token>` header for API authentication                                            |
| **JWT**            | JSON Web Token; a compact, self-contained token format for claims (e.g., user identity)                                      |
| **HS256**          | HMAC with SHA-256; a symmetric signing algorithm for JWTs                                                                    |
| **Sliding Window** | A rate-limiting algorithm that tracks request timestamps and evicts old ones as time progresses                              |
| **Request ID**     | A unique identifier assigned to each request for tracing and debugging                                                       |
| **ReAct Loop**     | Reasoning + Acting loop; the agent iterates between thinking about the problem and taking actions (e.g., retrieving sources) |
| **Reranking**      | Re-scoring and re-ordering search results by relevance using a specialized model                                             |
| **Session ID**     | A user-provided or auto-generated identifier for grouping related requests; enables memory continuity                        |

---

## Support and Debugging

### Identifying Issues

1. **Check the request ID:** Every request and response includes an `X-Request-ID` header
2. **Review logs:** Search server logs by request ID to trace the full request lifecycle
3. **Test the health endpoint:** `GET /health` confirms all dependencies are reachable
4. **Verify authentication:** Ensure the JWT is valid, not expired, and contains a `sub` claim

### Common Issues

| Issue                   | Cause                     | Solution                                           |
| ----------------------- | ------------------------- | -------------------------------------------------- |
| 401 Unauthorized        | Missing or invalid token  | Regenerate a valid JWT with a `sub` claim          |
| 429 Too Many Requests   | Rate limit exceeded       | Wait 60 seconds or use a different IP              |
| 503 Service Unavailable | Agent not initialized     | Retry after a few seconds; check startup logs      |
| 500 Internal Error      | Unexpected exception      | Check server logs using the request ID             |
| Empty citations         | No relevant sources found | Reformulate the query or increase `max_iterations` |

### Reporting Bugs

Include the following information:

1. Request ID (`X-Request-ID` header)
2. Timestamp of the request
3. Full request body (redact any secrets)
4. HTTP status code and response body
5. Server logs from the corresponding timestamp
