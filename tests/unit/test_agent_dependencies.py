"""Unit tests for AgentDependencies.

TDD RED phase — these tests are written before the implementation exists.

AgentDependencies is a frozen dataclass that acts as the single wiring point
for all protocol-typed components.  Its __post_init__ validates that each
slot satisfies its protocol at construction time, providing clear TypeError
messages rather than cryptic AttributeErrors deep inside the graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from research_agent.agent.dependencies import AgentDependencies

# ---------------------------------------------------------------------------
# Mock helpers — each produces a MagicMock that satisfies its protocol via
# __getattr__, so isinstance() checks on @runtime_checkable protocols pass.
# ---------------------------------------------------------------------------


def _mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.complete = MagicMock()
    return mock


def _mock_retriever() -> MagicMock:
    mock = MagicMock()
    mock.retrieve = MagicMock()
    return mock


def _mock_reranker() -> MagicMock:
    mock = MagicMock()
    mock.rerank = MagicMock()
    return mock


def _mock_memory() -> MagicMock:
    mock = MagicMock()
    mock.add = MagicMock()
    mock.search = MagicMock()
    return mock


def _mock_tool() -> MagicMock:
    mock = MagicMock()
    mock.name = "mock_tool"
    mock.description = "A mock tool"
    mock.execute = MagicMock()
    return mock


def _valid_deps(**overrides: object) -> AgentDependencies:
    """Return a valid AgentDependencies, optionally overriding specific fields."""
    defaults: dict[str, object] = {
        "llm_client": _mock_llm(),
        "retriever": _mock_retriever(),
        "reranker": _mock_reranker(),
        "memory_service": _mock_memory(),
        "tools": (_mock_tool(),),
    }
    defaults.update(overrides)
    return AgentDependencies(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction — valid cases
# ---------------------------------------------------------------------------


class TestAgentDependenciesConstruction:
    def test_valid_construction_succeeds(self) -> None:
        deps = _valid_deps()
        assert deps is not None

    def test_llm_client_field_accessible(self) -> None:
        llm = _mock_llm()
        deps = _valid_deps(llm_client=llm)
        assert deps.llm_client is llm

    def test_retriever_field_accessible(self) -> None:
        retriever = _mock_retriever()
        deps = _valid_deps(retriever=retriever)
        assert deps.retriever is retriever

    def test_reranker_field_accessible(self) -> None:
        reranker = _mock_reranker()
        deps = _valid_deps(reranker=reranker)
        assert deps.reranker is reranker

    def test_memory_service_field_accessible(self) -> None:
        memory = _mock_memory()
        deps = _valid_deps(memory_service=memory)
        assert deps.memory_service is memory

    def test_tools_field_accessible(self) -> None:
        tool = _mock_tool()
        deps = _valid_deps(tools=(tool,))
        assert deps.tools == (tool,)

    def test_tools_is_tuple(self) -> None:
        deps = _valid_deps()
        assert isinstance(deps.tools, tuple)

    def test_empty_tools_tuple_allowed(self) -> None:
        deps = _valid_deps(tools=())
        assert deps.tools == ()

    def test_multiple_tools_stored(self) -> None:
        t1 = _mock_tool()
        t2 = _mock_tool()
        t3 = _mock_tool()
        deps = _valid_deps(tools=(t1, t2, t3))
        assert deps.tools == (t1, t2, t3)
        assert len(deps.tools) == 3


# ---------------------------------------------------------------------------
# Immutability — frozen dataclass must reject field reassignment
# ---------------------------------------------------------------------------


class TestAgentDependenciesImmutability:
    def test_llm_client_is_frozen(self) -> None:
        deps = _valid_deps()
        with pytest.raises((AttributeError, TypeError)):
            deps.llm_client = _mock_llm()  # type: ignore[misc]

    def test_retriever_is_frozen(self) -> None:
        deps = _valid_deps()
        with pytest.raises((AttributeError, TypeError)):
            deps.retriever = _mock_retriever()  # type: ignore[misc]

    def test_tools_is_frozen(self) -> None:
        deps = _valid_deps()
        with pytest.raises((AttributeError, TypeError)):
            deps.tools = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol validation — __post_init__ must raise TypeError for invalid values
# ---------------------------------------------------------------------------


class TestAgentDependenciesProtocolValidation:
    def test_invalid_llm_client_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="llm_client"):
            AgentDependencies(
                llm_client=object(),  # type: ignore[arg-type]
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_invalid_retriever_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="retriever"):
            AgentDependencies(
                llm_client=_mock_llm(),
                retriever=object(),  # type: ignore[arg-type]
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_invalid_reranker_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="reranker"):
            AgentDependencies(
                llm_client=_mock_llm(),
                retriever=_mock_retriever(),
                reranker=object(),  # type: ignore[arg-type]
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_invalid_memory_service_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="memory_service"):
            AgentDependencies(
                llm_client=_mock_llm(),
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=object(),  # type: ignore[arg-type]
                tools=(),
            )

    def test_error_message_includes_protocol_name(self) -> None:
        with pytest.raises(TypeError, match="LLMClient"):
            AgentDependencies(
                llm_client=object(),  # type: ignore[arg-type]
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_error_message_includes_actual_type(self) -> None:
        with pytest.raises(TypeError, match="object"):
            AgentDependencies(
                llm_client=object(),  # type: ignore[arg-type]
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_none_llm_client_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="llm_client"):
            AgentDependencies(
                llm_client=None,  # type: ignore[arg-type]
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_none_retriever_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="retriever"):
            AgentDependencies(
                llm_client=_mock_llm(),
                retriever=None,  # type: ignore[arg-type]
                reranker=_mock_reranker(),
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_none_reranker_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="reranker"):
            AgentDependencies(
                llm_client=_mock_llm(),
                retriever=_mock_retriever(),
                reranker=None,  # type: ignore[arg-type]
                memory_service=_mock_memory(),
                tools=(),
            )

    def test_none_memory_service_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="memory_service"):
            AgentDependencies(
                llm_client=_mock_llm(),
                retriever=_mock_retriever(),
                reranker=_mock_reranker(),
                memory_service=None,  # type: ignore[arg-type]
                tools=(),
            )
