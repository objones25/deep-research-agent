"""Unit tests for tools/protocols.py — value objects and the Tool protocol.

TDD Phase: RED — these tests are written before any implementation exists.
Run: uv run pytest tests/unit/test_tool_protocols.py -v
"""

from __future__ import annotations

import dataclasses

import pytest

from research_agent.tools.protocols import (
    ScrapeInput,
    SearchInput,
    Tool,
    ToolExecutionError,
    ToolInput,
    ToolResult,
)

# ---------------------------------------------------------------------------
# ToolInput
# ---------------------------------------------------------------------------


class TestToolInput:
    def test_is_a_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ToolInput)

    def test_is_frozen(self) -> None:
        # ToolInput itself has no fields, but the frozen flag must be set so
        # subclasses that inherit it are also frozen by default.
        assert ToolInput.__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_search_input_is_subclass(self) -> None:
        assert issubclass(SearchInput, ToolInput)

    def test_scrape_input_is_subclass(self) -> None:
        assert issubclass(ScrapeInput, ToolInput)

    def test_search_input_instance_is_tool_input(self) -> None:
        assert isinstance(SearchInput(query="test"), ToolInput)

    def test_scrape_input_instance_is_tool_input(self) -> None:
        assert isinstance(ScrapeInput(url="https://example.com"), ToolInput)


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_result_defaults(self) -> None:
        result = ToolResult(is_error=False, content="findings")
        assert result.is_error is False
        assert result.content == "findings"
        assert result.error is None
        assert result.metadata == {}

    def test_error_result_defaults(self) -> None:
        result = ToolResult(is_error=True, error="rate limit hit")
        assert result.is_error is True
        assert result.error == "rate limit hit"
        assert result.content is None

    def test_custom_metadata(self) -> None:
        result = ToolResult(is_error=False, content="x", metadata={"url": "https://a.com"})
        assert result.metadata == {"url": "https://a.com"}

    def test_is_frozen(self) -> None:
        result = ToolResult(is_error=False, content="x")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.is_error = True  # type: ignore[misc]

    def test_metadata_default_not_shared_between_instances(self) -> None:
        r1 = ToolResult(is_error=False, content="a")
        r2 = ToolResult(is_error=False, content="b")
        assert r1.metadata is not r2.metadata

    def test_equality_based_on_values(self) -> None:
        r1 = ToolResult(is_error=False, content="same")
        r2 = ToolResult(is_error=False, content="same")
        assert r1 == r2


# ---------------------------------------------------------------------------
# SearchInput
# ---------------------------------------------------------------------------


class TestSearchInput:
    def test_valid_construction_default_limit(self) -> None:
        inp = SearchInput(query="climate change research")
        assert inp.query == "climate change research"
        assert inp.limit == 5

    def test_valid_construction_custom_limit(self) -> None:
        inp = SearchInput(query="AI safety", limit=10)
        assert inp.limit == 10

    def test_empty_query_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="query"):
            SearchInput(query="")

    def test_whitespace_only_query_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="query"):
            SearchInput(query="   ")

    def test_tab_only_query_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="query"):
            SearchInput(query="\t\n")

    def test_zero_limit_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            SearchInput(query="valid query", limit=0)

    def test_negative_limit_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            SearchInput(query="valid query", limit=-3)

    def test_is_frozen(self) -> None:
        inp = SearchInput(query="test")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            inp.query = "mutated"  # type: ignore[misc]

    def test_equality_based_on_values(self) -> None:
        assert SearchInput(query="same", limit=5) == SearchInput(query="same", limit=5)
        assert SearchInput(query="a") != SearchInput(query="b")


# ---------------------------------------------------------------------------
# ScrapeInput
# ---------------------------------------------------------------------------


class TestScrapeInput:
    def test_valid_https_url(self) -> None:
        inp = ScrapeInput(url="https://example.com/article")
        assert inp.url == "https://example.com/article"
        assert inp.only_main_content is True

    def test_valid_http_url(self) -> None:
        inp = ScrapeInput(url="http://example.com")
        assert inp.url == "http://example.com"

    def test_only_main_content_false(self) -> None:
        inp = ScrapeInput(url="https://example.com", only_main_content=False)
        assert inp.only_main_content is False

    def test_empty_url_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="url"):
            ScrapeInput(url="")

    def test_ftp_scheme_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="url"):
            ScrapeInput(url="ftp://example.com/file.txt")

    def test_no_scheme_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="url"):
            ScrapeInput(url="example.com/page")

    def test_plain_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="url"):
            ScrapeInput(url="not a url at all")

    def test_scheme_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="url"):
            ScrapeInput(url="https://")

    def test_is_frozen(self) -> None:
        inp = ScrapeInput(url="https://example.com")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            inp.url = "https://other.com"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolExecutionError
# ---------------------------------------------------------------------------


class TestToolExecutionError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(ToolExecutionError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ToolExecutionError, match="MCP call failed"):
            raise ToolExecutionError("MCP call failed")

    def test_message_preserved(self) -> None:
        err = ToolExecutionError("network timeout after 30s")
        assert str(err) == "network timeout after 30s"

    def test_supports_chained_exceptions(self) -> None:
        cause = RuntimeError("connection refused")
        err = ToolExecutionError("wrapped")
        err.__cause__ = cause
        assert err.__cause__ is cause


# ---------------------------------------------------------------------------
# Tool protocol structural compliance
# ---------------------------------------------------------------------------


class TestToolProtocol:
    def test_compliant_class_passes_isinstance(self) -> None:
        class GoodTool:
            @property
            def name(self) -> str:
                return "good_tool"

            @property
            def description(self) -> str:
                return "Does something useful."

            async def execute(self, tool_input: ToolInput) -> ToolResult:
                return ToolResult(is_error=False, content="result")

        assert isinstance(GoodTool(), Tool)

    def test_missing_name_fails_isinstance(self) -> None:
        class NoName:
            @property
            def description(self) -> str:
                return "no name"

            async def execute(self, tool_input: ToolInput) -> ToolResult:
                return ToolResult(is_error=False, content="x")

        assert not isinstance(NoName(), Tool)

    def test_missing_description_fails_isinstance(self) -> None:
        class NoDescription:
            @property
            def name(self) -> str:
                return "no_desc"

            async def execute(self, tool_input: ToolInput) -> ToolResult:
                return ToolResult(is_error=False, content="x")

        assert not isinstance(NoDescription(), Tool)

    def test_missing_execute_fails_isinstance(self) -> None:
        class NoExecute:
            @property
            def name(self) -> str:
                return "no_exec"

            @property
            def description(self) -> str:
                return "Missing execute"

        assert not isinstance(NoExecute(), Tool)

    def test_all_members_missing_fails_isinstance(self) -> None:
        assert not isinstance(object(), Tool)
