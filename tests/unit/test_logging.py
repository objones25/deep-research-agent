"""Tests for the structured logging configuration module.

All tests use structlog.testing.capture_logs() which temporarily replaces the
processor chain with a capture processor, giving access to the raw event dict
before rendering.  contextvars are automatically merged by capture_logs (>= 21.5).
"""

from __future__ import annotations

import pytest
import structlog.testing

from research_agent.logging import (
    bind_contextvars,
    clear_contextvars,
    configure_logging,
    get_logger,
)


@pytest.mark.unit
class TestConfigureLogging:
    def test_json_mode_does_not_raise(self) -> None:
        configure_logging(log_level="INFO", log_json=True)

    def test_console_mode_does_not_raise(self) -> None:
        configure_logging(log_level="DEBUG", log_json=False)

    def test_all_standard_log_levels_accepted(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            configure_logging(log_level=level, log_json=False)

    def test_default_args_do_not_raise(self) -> None:
        # configure_logging() must be callable with zero arguments
        configure_logging()


@pytest.mark.unit
class TestGetLogger:
    def test_returns_non_none_logger(self) -> None:
        logger = get_logger("research_agent.test")
        assert logger is not None

    def test_get_logger_without_name_does_not_raise(self) -> None:
        logger = get_logger()
        assert logger is not None

    def test_info_event_captured(self) -> None:
        with structlog.testing.capture_logs() as cap:
            get_logger("test").info("hello_world", key="value")

        assert len(cap) == 1
        assert cap[0]["event"] == "hello_world"
        assert cap[0]["key"] == "value"
        assert cap[0]["log_level"] == "info"

    def test_error_event_captured(self) -> None:
        with structlog.testing.capture_logs() as cap:
            get_logger("test").error("something_failed", reason="oops")

        assert cap[0]["event"] == "something_failed"
        assert cap[0]["log_level"] == "error"

    def test_debug_event_captured(self) -> None:
        with structlog.testing.capture_logs() as cap:
            get_logger("test").debug("debug_event", detail="x")

        assert cap[0]["event"] == "debug_event"
        assert cap[0]["log_level"] == "debug"

    def test_warning_event_captured(self) -> None:
        with structlog.testing.capture_logs() as cap:
            get_logger("test").warning("warn_event")

        assert cap[0]["event"] == "warn_event"
        assert cap[0]["log_level"] == "warning"

    def test_multiple_key_value_pairs_captured(self) -> None:
        with structlog.testing.capture_logs() as cap:
            get_logger("test").info(
                "retrieve_complete",
                num_results=20,
                latency_ms=111.5,
                top_score=0.92,
            )

        assert cap[0]["num_results"] == 20
        assert cap[0]["latency_ms"] == 111.5
        assert cap[0]["top_score"] == 0.92

    def test_different_names_produce_independent_loggers(self) -> None:
        logger_a = get_logger("research_agent.retrieval")
        logger_b = get_logger("research_agent.llm")

        with structlog.testing.capture_logs() as cap:
            logger_a.info("from_a")
            logger_b.info("from_b")

        assert cap[0]["event"] == "from_a"
        assert cap[1]["event"] == "from_b"


@pytest.mark.unit
class TestContextvars:
    def setup_method(self) -> None:
        clear_contextvars()

    def teardown_method(self) -> None:
        clear_contextvars()

    def test_bind_request_id_appears_in_log_event(self) -> None:
        bind_contextvars(request_id="req_abc123")

        with structlog.testing.capture_logs([structlog.contextvars.merge_contextvars]) as cap:
            get_logger("test").info("event")

        assert cap[0]["request_id"] == "req_abc123"

    def test_clear_removes_bound_request_id(self) -> None:
        bind_contextvars(request_id="req_xyz")
        clear_contextvars()

        with structlog.testing.capture_logs([structlog.contextvars.merge_contextvars]) as cap:
            get_logger("test").info("event")

        assert "request_id" not in cap[0]

    def test_multiple_context_vars_all_present(self) -> None:
        bind_contextvars(request_id="req_001", user_id="user_42")

        with structlog.testing.capture_logs([structlog.contextvars.merge_contextvars]) as cap:
            get_logger("test").info("event")

        assert cap[0]["request_id"] == "req_001"
        assert cap[0]["user_id"] == "user_42"

    def test_context_vars_persist_across_multiple_log_calls(self) -> None:
        bind_contextvars(request_id="req_persist")

        with structlog.testing.capture_logs([structlog.contextvars.merge_contextvars]) as cap:
            get_logger("test").info("first")
            get_logger("test").info("second")

        assert cap[0]["request_id"] == "req_persist"
        assert cap[1]["request_id"] == "req_persist"

    def test_bind_contextvars_is_importable_from_module(self) -> None:
        # Verify re-export works — callable with keyword args
        bind_contextvars(foo="bar")
        with structlog.testing.capture_logs([structlog.contextvars.merge_contextvars]) as cap:
            get_logger("test").info("e")
        assert cap[0]["foo"] == "bar"

    def test_clear_contextvars_is_importable_from_module(self) -> None:
        # Verify re-export works — callable with no args
        clear_contextvars()
