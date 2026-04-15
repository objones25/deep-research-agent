"""Tests for HuggingFaceClient LLM implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_agent.llm.huggingface import HuggingFaceClient
from research_agent.llm.protocols import LLMClient, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(
    *,
    content: str | None = "test response",
    model: str = "test-model",
    max_tokens: int = 512,
) -> tuple[HuggingFaceClient, AsyncMock]:
    """Return a HuggingFaceClient whose chat.completions.create returns *content*."""
    hf_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    hf_client.chat.completions.create = AsyncMock(return_value=mock_response)

    client = HuggingFaceClient(client=hf_client, model=model, max_tokens=max_tokens)
    return client, hf_client


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceClientInputValidation:
    async def test_raises_on_empty_messages(self) -> None:
        client, _ = make_client()
        with pytest.raises(ValueError, match="empty"):
            await client.complete([])

    async def test_raises_when_api_returns_none_content(self) -> None:
        client, _ = make_client(content=None)
        with pytest.raises(ValueError, match="no content"):
            await client.complete([Message(role="user", content="hello")])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceClientHappyPath:
    async def test_returns_string_from_api_response(self) -> None:
        client, _ = make_client(content="Paris is the capital of France.")
        result = await client.complete(
            [Message(role="user", content="What is the capital of France?")]
        )
        assert result == "Paris is the capital of France."

    async def test_returns_string_type(self) -> None:
        client, _ = make_client(content="hello")
        result = await client.complete([Message(role="user", content="ping")])
        assert isinstance(result, str)

    async def test_system_and_user_messages_both_forwarded(self) -> None:
        client, hf_client = make_client()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is 2+2?"),
        ]
        await client.complete(messages)

        _, kwargs = hf_client.chat.completions.create.call_args
        forwarded = kwargs["messages"]
        assert forwarded == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

    async def test_assistant_message_forwarded_in_multi_turn(self) -> None:
        client, hf_client = make_client()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        await client.complete(messages)

        _, kwargs = hf_client.chat.completions.create.call_args
        assert kwargs["messages"][1] == {"role": "assistant", "content": "Hi there!"}

    async def test_messages_order_preserved(self) -> None:
        client, hf_client = make_client()
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="u2"),
        ]
        await client.complete(messages)

        _, kwargs = hf_client.chat.completions.create.call_args
        roles = [m["role"] for m in kwargs["messages"]]
        assert roles == ["system", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# API call parameter forwarding
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceClientAPICallForwarding:
    async def test_model_passed_to_create(self) -> None:
        client, hf_client = make_client(model="Qwen/Qwen3-32B")
        await client.complete([Message(role="user", content="hello")])

        _, kwargs = hf_client.chat.completions.create.call_args
        assert kwargs["model"] == "Qwen/Qwen3-32B"

    async def test_max_tokens_from_constructor_passed_to_create(self) -> None:
        client, hf_client = make_client(max_tokens=2048)
        await client.complete([Message(role="user", content="hello")])

        _, kwargs = hf_client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 2048

    async def test_create_called_exactly_once_per_complete(self) -> None:
        client, hf_client = make_client()
        await client.complete([Message(role="user", content="hello")])

        hf_client.chat.completions.create.assert_called_once()

    async def test_single_user_message_forwarded_as_dict(self) -> None:
        client, hf_client = make_client()
        await client.complete([Message(role="user", content="hello world")])

        _, kwargs = hf_client.chat.completions.create.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "hello world"}]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceClientProtocolConformance:
    def test_satisfies_llm_client_protocol(self) -> None:
        client, _ = make_client()
        assert isinstance(client, LLMClient)
