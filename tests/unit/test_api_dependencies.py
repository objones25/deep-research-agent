"""Unit tests for research_agent.api.dependencies — AgentRunner protocol."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.datastructures import Address

from research_agent.api.dependencies import AgentRunner, check_rate_limit
from research_agent.models.research import ResearchQuery, ResearchReport

# ---------------------------------------------------------------------------
# AgentRunner protocol
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentRunnerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        # isinstance checks against runtime_checkable Protocol must not raise
        class ConcreteRunner:
            async def run(self, query: ResearchQuery) -> ResearchReport:
                raise NotImplementedError

        runner = ConcreteRunner()
        assert isinstance(runner, AgentRunner)

    def test_missing_run_method_does_not_satisfy_protocol(self) -> None:
        class BrokenRunner:
            pass

        assert not isinstance(BrokenRunner(), AgentRunner)

    def test_wrong_method_name_does_not_satisfy_protocol(self) -> None:
        class WrongRunner:
            async def execute(self, query: ResearchQuery) -> ResearchReport:
                raise NotImplementedError

        assert not isinstance(WrongRunner(), AgentRunner)


# ---------------------------------------------------------------------------
# check_rate_limit dependency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckRateLimitDependency:
    async def test_passes_for_known_client_ip(self) -> None:
        """check_rate_limit does not raise for a first request from a known IP."""
        mock_request = MagicMock()
        mock_request.client = Address(host="192.0.2.1", port=9999)
        # Should not raise — first request is well within the 60/60s limit
        await check_rate_limit(mock_request)

    async def test_uses_unknown_when_client_is_none(self) -> None:
        """check_rate_limit falls back to 'unknown' when request.client is None."""
        mock_request = MagicMock()
        mock_request.client = None
        # Should not raise — first request for 'unknown' is within limit
        await check_rate_limit(mock_request)
