"""HuggingFace-backed implementation of the LLMClient protocol.

Routes chat-completion inference through Featherless AI via the
HuggingFace Hub ``AsyncInferenceClient`` (OpenAI-compatible endpoint).

Usage::

    from huggingface_hub import AsyncInferenceClient
    from research_agent.llm import HuggingFaceClient
    from research_agent.config import get_settings

    settings = get_settings()
    hf_client = AsyncInferenceClient(
        provider="featherless-ai",
        api_key=settings.hf_token.get_secret_value(),
    )
    llm = HuggingFaceClient(
        client=hf_client,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )
"""

from __future__ import annotations

import time

from huggingface_hub import AsyncInferenceClient

from research_agent.llm.protocols import Message
from research_agent.logging import get_logger

_log = get_logger(__name__)


class HuggingFaceClient:
    """LLM client backed by HuggingFace Hub's async inference API.

    Satisfies the ``LLMClient`` protocol.

    Parameters
    ----------
    client:
        An initialised ``AsyncInferenceClient``.  Inject this rather than
        constructing it internally so the dependency is testable and the
        provider/api_key are configured at the call site.
    model:
        HuggingFace model ID (e.g. ``"Qwen/Qwen3-32B"``).
    max_tokens:
        Maximum tokens to generate per completion.  Configured once at
        construction time; applied to every ``complete`` call.
    """

    def __init__(
        self,
        client: AsyncInferenceClient,
        model: str,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, messages: list[Message]) -> str:
        """Return the model's text response to *messages*.

        Parameters
        ----------
        messages:
            Ordered conversation turns.  Must not be empty.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        ValueError
            If *messages* is empty or the model returns no content.
        """
        if not messages:
            raise ValueError("messages must not be empty.")

        _log.info("llm_request_started", model=self._model, num_messages=len(messages))
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=self._max_tokens,
        )

        content: str | None = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned no content; check model availability and quota.")
        _log.info(
            "llm_request_complete",
            model=self._model,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            response_len=len(content),
        )
        return content


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

from research_agent.config import Settings  # noqa: E402
from research_agent.llm.registry import llm_registry  # noqa: E402


@llm_registry.register("huggingface")
async def _build_huggingface(settings: Settings) -> HuggingFaceClient:
    import huggingface_hub

    hf_client = huggingface_hub.AsyncInferenceClient(
        provider="featherless-ai",
        api_key=settings.hf_token.get_secret_value(),
    )
    return HuggingFaceClient(
        client=hf_client,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )
