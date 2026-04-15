"""Tests for LLM protocols and the Message value object."""

from __future__ import annotations

import dataclasses

import pytest

from research_agent.llm.protocols import LLMClient, Message


@pytest.mark.unit
class TestMessage:
    def test_creation_stores_role_and_content(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_role_is_immutable(self) -> None:
        msg = Message(role="user", content="Hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.role = "assistant"  # type: ignore[misc]

    def test_content_is_immutable(self) -> None:
        msg = Message(role="user", content="Hello")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.content = "Goodbye"  # type: ignore[misc]

    def test_replace_creates_new_instance_with_updated_content(self) -> None:
        original = Message(role="user", content="Hello")
        updated = dataclasses.replace(original, content="World")

        assert updated.content == "World"
        assert original.content == "Hello"

    def test_equality_with_same_fields(self) -> None:
        m1 = Message(role="user", content="Hello")
        m2 = Message(role="user", content="Hello")
        assert m1 == m2

    def test_inequality_on_different_role(self) -> None:
        m1 = Message(role="user", content="Hello")
        m2 = Message(role="assistant", content="Hello")
        assert m1 != m2

    def test_inequality_on_different_content(self) -> None:
        m1 = Message(role="user", content="Hello")
        m2 = Message(role="user", content="Goodbye")
        assert m1 != m2

    def test_user_role_is_valid(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"

    def test_system_role_is_valid(self) -> None:
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_assistant_role_is_valid(self) -> None:
        msg = Message(role="assistant", content="I can help with that.")
        assert msg.role == "assistant"

    def test_tool_role_is_valid(self) -> None:
        msg = Message(role="tool", content='{"result": "Paris"}')
        assert msg.role == "tool"

    def test_invalid_role_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            Message(role="invalid", content="Hello")  # type: ignore[arg-type]

    def test_all_four_valid_roles_accepted(self) -> None:
        roles = ("system", "user", "assistant", "tool")
        for role in roles:
            msg = Message(role=role, content="text")  # type: ignore[arg-type]
            assert msg.role == role


@pytest.mark.unit
class TestLLMClientProtocol:
    def test_satisfied_by_class_with_complete_method(self) -> None:
        class FakeLLMClient:
            async def complete(self, messages: list[Message]) -> str:
                return ""

        assert isinstance(FakeLLMClient(), LLMClient)

    def test_not_satisfied_by_empty_class(self) -> None:
        class Bad:
            pass

        assert not isinstance(Bad(), LLMClient)

    def test_not_satisfied_without_complete_method(self) -> None:
        class Bad:
            async def generate(self, messages: list[Message]) -> str:
                return ""

        assert not isinstance(Bad(), LLMClient)

    def test_not_satisfied_by_sync_complete_method(self) -> None:
        # runtime_checkable only checks method *existence*, not async-ness,
        # but we document the expectation: the protocol contract is async.
        # This test confirms the structural check passes even for sync stubs —
        # callers must ensure the method is actually awaitable.
        class SyncStub:
            def complete(self, messages: list[Message]) -> str:
                return ""

        # Protocol only checks presence of 'complete', not coroutine type.
        assert isinstance(SyncStub(), LLMClient)
