"""Unit tests for the MemoryService protocol.

These tests verify structural correctness of the protocol itself — they
do not test any concrete implementation.  All tests are synchronous and
require no I/O.
"""

from __future__ import annotations

import pytest

from research_agent.memory.protocols import MemoryService

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _ValidStub:
    """Minimal class that satisfies MemoryService."""

    async def add(self, session_id: str, content: str) -> None:
        pass

    async def search(self, session_id: str, query: str) -> list[str]:
        return []


class _MissingAddStub:
    """Stub that only implements search — missing add."""

    async def search(self, session_id: str, query: str) -> list[str]:
        return []


class _MissingSearchStub:
    """Stub that only implements add — missing search."""

    async def add(self, session_id: str, content: str) -> None:
        pass


class _EmptyStub:
    """Stub with neither method."""

    pass


# ---------------------------------------------------------------------------
# Protocol runtime-checkable tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryServiceProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """MemoryService can be used in isinstance checks."""
        stub = _ValidStub()
        # Should not raise TypeError
        result = isinstance(stub, MemoryService)
        assert result is True

    def test_valid_implementation_satisfies_protocol(self) -> None:
        stub = _ValidStub()
        assert isinstance(stub, MemoryService)

    def test_missing_add_fails_protocol(self) -> None:
        stub = _MissingAddStub()
        assert not isinstance(stub, MemoryService)

    def test_missing_search_fails_protocol(self) -> None:
        stub = _MissingSearchStub()
        assert not isinstance(stub, MemoryService)

    def test_empty_class_fails_protocol(self) -> None:
        stub = _EmptyStub()
        assert not isinstance(stub, MemoryService)

    def test_protocol_has_add_method(self) -> None:
        assert hasattr(MemoryService, "add")

    def test_protocol_has_search_method(self) -> None:
        assert hasattr(MemoryService, "search")
