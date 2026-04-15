FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies (no dev extras, no editable install)
RUN uv sync --frozen --no-dev --no-editable

# Copy application source
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Use --factory so create_app() is called at startup with real env vars,
# not at import time where required secrets are absent.
CMD ["uvicorn", "research_agent.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
