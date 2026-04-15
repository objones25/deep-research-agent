#!/usr/bin/env python3
"""Smoke test the running API (local or Railway).

Usage:
    # Against local dev server (default)
    uv run python scripts/smoke_test.py

    # Against a deployed Railway instance
    uv run python scripts/smoke_test.py --url https://<your-railway-host>

The script:
  1. Loads SECRET_KEY from .env (via pydantic Settings)
  2. Mints a short-lived HS256 JWT
  3. Hits GET /health and asserts 200
  4. Hits POST /research with a real query and pretty-prints the report
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx
import jwt


def _make_token(secret: str, algorithm: str = "HS256") -> str:
    payload = {
        "sub": "smoke-test@example.com",
        "exp": int(time.time()) + 300,  # 5-minute token
    }
    return jwt.encode(payload, secret, algorithm=algorithm)


def _check_health(base_url: str) -> None:
    print(f"[health]  GET {base_url}/health")
    r = httpx.get(f"{base_url}/health", timeout=10)
    print(f"          status={r.status_code}")
    print(f"          body={json.dumps(r.json(), indent=2)}")
    if r.status_code != 200:
        print("[FAIL]  /health returned non-200", file=sys.stderr)
        sys.exit(1)
    print("[OK]")


def _run_research(base_url: str, token: str, query: str) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "session_id": "smoke-test-session",
        "max_iterations": 3,
    }
    print(f"\n[research] POST {base_url}/research")
    print(f"           query={query!r}")
    r = httpx.post(
        f"{base_url}/research",
        headers=headers,
        json=payload,
        timeout=600,  # agent may take several minutes (multiple LLM calls ~60s each)
    )
    print(f"           status={r.status_code}")
    if r.status_code != 200:
        print(f"[FAIL]  /research returned {r.status_code}", file=sys.stderr)
        print(r.text, file=sys.stderr)
        sys.exit(1)
    body = r.json()
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(body.get("summary", "(no summary)"))
    print(f"\n{'=' * 60}")
    print(f"CITATIONS ({len(body.get('citations', []))} total)")
    print("=" * 60)
    for i, c in enumerate(body.get("citations", []), 1):
        print(f"  [{i}] {c['url']}  (score={c['relevance_score']:.2f})")
        print(f"       {c['content_snippet'][:120]!r}")
    print(f"\nsession_id: {body.get('session_id')}")
    print("[OK]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Deep Research Agent API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--query",
        default="What are the key advances in retrieval-augmented generation in 2024?",
        help="Research query to submit",
    )
    args = parser.parse_args()

    # Load SECRET_KEY via pydantic Settings (reads .env automatically)
    from research_agent.config import get_settings

    settings = get_settings()
    secret = settings.secret_key.get_secret_value()
    algorithm = settings.jwt_algorithm

    token = _make_token(secret, algorithm)

    _check_health(args.url)
    _run_research(args.url, token, args.query)


if __name__ == "__main__":
    main()
