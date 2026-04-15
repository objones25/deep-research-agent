"""Structured logging configuration and helpers.

Call :func:`configure_logging` once at application startup, then use
:func:`get_logger` everywhere in place of ``import logging``.

Usage::

    from research_agent.logging import configure_logging, get_logger

    configure_logging(log_level="INFO", log_json=True)   # once, at startup
    log = get_logger(__name__)
    log.info("retrieve_complete", num_results=20, latency_ms=111.5)

Request-ID context (async-safe via contextvars)::

    from research_agent.logging import bind_contextvars, clear_contextvars

    bind_contextvars(request_id="req_abc123")
    # Every log call in this coroutine now includes request_id automatically.
    clear_contextvars()   # call in middleware teardown

Output:
- ``log_json=True``  → one JSON object per line, suitable for Railway / log
  aggregators.
- ``log_json=False`` → coloured console output for local development.
"""

from __future__ import annotations

import sys

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_contextvars",
    "clear_contextvars",
]

# Shared processors run regardless of output format.
_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    # Merge any values bound via bind_contextvars() into every event dict.
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
]


def configure_logging(log_level: str = "INFO", log_json: bool = False) -> None:
    """Configure structlog for the whole application.

    Must be called once before the first log statement.  Calling it again
    (e.g., in tests) is safe — it simply reconfigures the pipeline.

    Args:
        log_level: Minimum severity to emit. One of DEBUG / INFO / WARNING /
                   ERROR / CRITICAL (case-insensitive).
        log_json:  Render log lines as JSON when *True* (production / Railway).
                   Render coloured human-readable output when *False* (dev).
    """
    if log_json:
        processors: list[structlog.types.Processor] = _SHARED_PROCESSORS + [
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = _SHARED_PROCESSORS + [
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        # Keep False so tests can call configure_logging() multiple times.
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a named :class:`structlog.stdlib.BoundLogger`.

    Args:
        name: Logger name, typically ``__name__``.  Pass *None* to use an
              anonymous logger.

    Returns:
        A bound logger that accepts ``info()``, ``debug()``, ``warning()``,
        ``error()``, and ``exception()`` calls with arbitrary keyword context.
    """
    if name is not None:
        return structlog.stdlib.get_logger(name)
    return structlog.stdlib.get_logger()
