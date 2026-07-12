"""OpenAI-compatible LLM provider.

Implements the same `LLMProvider` interface as `GroqProvider`, so the
rest of the application never knows which provider is active. Uses the
official `openai` SDK (v1+) which also works against any OpenAI-compatible
endpoint (Together AI, Anyscale, local vLLM, etc.) by pointing
`base_url` at the right host.

Not installed in requirements.txt by default (Groq is the primary
provider per Phase 1) — add `openai>=1.0.0` to requirements.txt if you
want to use this.
"""

import time

from app.config.settings import get_settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import LLMMessage, LLMProvider, LLMResponse

log = get_logger("llm.openai")


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible chat-completion provider.

    Usage with a local vLLM server:
        OpenAIProvider(base_url="http://localhost:8000/v1", api_key="none")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required to use OpenAIProvider. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc

        settings = get_settings()
        self._model = model or settings.openai_model
        self._timeout = settings.llm_request_timeout_seconds
        self._default_temperature = settings.llm_temperature
        self._default_max_tokens = settings.llm_max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url,
            timeout=self._timeout,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
        t0 = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=sdk_messages,
                temperature=temperature if temperature is not None else self._default_temperature,
                max_tokens=max_tokens or self._default_max_tokens,
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI API error: {exc}") from exc

        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage

        log.bind(
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=round(latency_ms, 1),
        ).info("LLM request complete")

        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model,
            latency_ms=latency_ms,
        )
