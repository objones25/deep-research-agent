"""Unit tests for Mem0MemoryService.

The AsyncMemoryClient is mocked at the boundary — no network calls are
made.  Every test follows the Arrange / Act / Assert pattern.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from structlog.testing import capture_logs

from research_agent.memory.mem0 import Mem0MemoryService
from research_agent.memory.protocols import MemoryService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a MagicMock whose async methods are AsyncMocks."""
    client = MagicMock()
    client.add = AsyncMock(return_value=None)
    client.search = AsyncMock(
        return_value=[
            {"id": "mem_1", "memory": "Paris is the capital of France", "score": 0.95},
            {"id": "mem_2", "memory": "The Eiffel Tower is in Paris", "score": 0.87},
        ]
    )
    return client


@pytest.fixture()
def service(mock_client: MagicMock) -> Mem0MemoryService:
    return Mem0MemoryService(client=mock_client)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMem0ServiceProtocolCompliance:
    def test_satisfies_memory_service_protocol(self, service: Mem0MemoryService) -> None:
        assert isinstance(service, MemoryService)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMem0ServiceAdd:
    async def test_add_calls_client_add(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        await service.add(session_id="sess_abc", content="some research context")
        mock_client.add.assert_awaited_once()

    async def test_add_passes_session_id_as_user_id(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        await service.add(session_id="sess_xyz", content="hello")
        _, kwargs = mock_client.add.call_args
        assert kwargs["user_id"] == "sess_xyz"

    async def test_add_wraps_content_as_user_message(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        await service.add(session_id="sess_1", content="the sky is blue")
        args, _ = mock_client.add.call_args
        messages = args[0]
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "the sky is blue"

    async def test_add_returns_none(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        result = await service.add(session_id="sess_1", content="x")
        assert result is None

    async def test_add_propagates_client_exception(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        mock_client.add.side_effect = RuntimeError("Mem0 unavailable")
        with pytest.raises(RuntimeError, match="Mem0 unavailable"):
            await service.add(session_id="sess_err", content="x")


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMem0ServiceSearch:
    async def test_search_calls_client_search(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        await service.search(session_id="sess_abc", query="capital cities")
        mock_client.search.assert_awaited_once()

    async def test_search_passes_query_as_first_arg(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        await service.search(session_id="sess_1", query="my query")
        args, _ = mock_client.search.call_args
        assert args[0] == "my query"

    async def test_search_passes_session_id_as_user_id(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        await service.search(session_id="sess_xyz", query="anything")
        _, kwargs = mock_client.search.call_args
        assert kwargs["user_id"] == "sess_xyz"

    async def test_search_returns_list_of_strings(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        results = await service.search(session_id="sess_1", query="Paris")
        assert isinstance(results, list)
        assert all(isinstance(item, str) for item in results)

    async def test_search_extracts_memory_field(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        results = await service.search(session_id="sess_1", query="Paris")
        assert results == [
            "Paris is the capital of France",
            "The Eiffel Tower is in Paris",
        ]

    async def test_search_returns_empty_list_when_no_results(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        mock_client.search.return_value = []
        results = await service.search(session_id="sess_1", query="obscure query")
        assert results == []

    async def test_search_ignores_entries_missing_memory_field(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        mock_client.search.return_value = [
            {"id": "mem_1", "memory": "valid entry", "score": 0.9},
            {"id": "mem_2", "score": 0.8},  # no "memory" key
        ]
        results = await service.search(session_id="sess_1", query="q")
        assert results == ["valid entry"]

    async def test_search_propagates_client_exception(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        mock_client.search.side_effect = RuntimeError("Mem0 unavailable")
        with pytest.raises(RuntimeError, match="Mem0 unavailable"):
            await service.search(session_id="sess_err", query="q")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMem0ServiceLogging:
    async def test_add_logs_memory_add_complete(self, service: Mem0MemoryService) -> None:
        with capture_logs() as cap:
            await service.add(session_id="sess-log", content="some content")
        events = [e["event"] for e in cap]
        assert "memory_add_complete" in events
        entry = next(e for e in cap if e["event"] == "memory_add_complete")
        assert entry["log_level"] == "info"
        assert entry["session_id"] == "sess-log"
        assert "latency_ms" in entry

    async def test_search_logs_memory_search_complete(self, service: Mem0MemoryService) -> None:
        with capture_logs() as cap:
            await service.search(session_id="sess-log", query="test query")
        events = [e["event"] for e in cap]
        assert "memory_search_complete" in events
        entry = next(e for e in cap if e["event"] == "memory_search_complete")
        assert entry["log_level"] == "info"
        assert entry["session_id"] == "sess-log"
        assert entry["num_results"] == 2
        assert "latency_ms" in entry
        assert entry["hit"] is True

    async def test_search_logs_hit_false_when_no_results(
        self, service: Mem0MemoryService, mock_client: MagicMock
    ) -> None:
        mock_client.search.return_value = []
        with capture_logs() as cap:
            await service.search(session_id="sess-log", query="nothing")
        entry = next(e for e in cap if e["event"] == "memory_search_complete")
        assert entry["hit"] is False
        assert entry["num_results"] == 0
