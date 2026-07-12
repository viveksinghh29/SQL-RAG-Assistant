"""Groq LLM provider — concrete implementation of LLMProvider.

Uses the official `groq` SDK. Handles:
- Async chat completion via `AsyncGroq`
- Exponential-backoff retry on transient errors (rate limits, 5xx)
- Timeout enforcement from settings
- Token usage tracking (logged per-request for cost monitoring)
- Translation of all Groq-specific exceptions into `LLMProviderError`
  so callers never depend on Groq internals

Default model: llama-3.3-70b-versatile (per Phase 1 decision).
"""

import time

from groq import AsyncGroq, APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import get_settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import LLMMessage, LLMProvider, LLMResponse

log = get_logger("llm.groq")


def _is_retryable(exc: BaseException) -> bool:
    """Retry on rate limits and transient server errors, not on bad requests."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    if isinstance(exc, APIConnectionError):
        return True
    return False


class GroqProvider(LLMProvider):
    """Async Groq chat-completion provider.

    Instantiate once per process (or once per request via DI — both are
    fine since `AsyncGroq` is lightweight). A single instance is thread/
    task-safe.
    """

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._model = model or settings.groq_model
        self._timeout = settings.llm_request_timeout_seconds
        self._default_temperature = settings.llm_temperature
        self._default_max_tokens = settings.llm_max_tokens
        self._client = AsyncGroq(api_key=api_key or settings.groq_api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request to Groq and return a normalized response.

        Retries up to 3 times with exponential back-off on rate limits and
        transient server errors. Non-retryable errors (bad API key, 4xx
        except 429) are surfaced immediately as `LLMProviderError`.
        """
        sdk_messages = [{"role": m.role, "content": m.content} for m in messages]
        t0 = time.monotonic()

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=sdk_messages,
                temperature=temperature if temperature is not None else self._default_temperature,
                max_tokens=max_tokens or self._default_max_tokens,
                timeout=self._timeout,
            )
        except RateLimitError as exc:
            log.warning(f"Groq rate limit hit: {exc}")
            raise  # tenacity will catch and retry
        except APIStatusError as exc:
            if exc.status_code >= 500:
                log.warning(f"Groq server error {exc.status_code}, will retry: {exc}")
                raise
            # 4xx (except 429 handled above) — non-retryable
            raise LLMProviderError(
                f"Groq API error {exc.status_code}: {exc.message}",
                details={"status_code": exc.status_code},
            ) from exc
        except APIConnectionError as exc:
            log.warning(f"Groq connection error, will retry: {exc}")
            raise
        except Exception as exc:
            raise LLMProviderError(f"Unexpected Groq error: {exc}") from exc

        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage

        log.bind(
            model=self._model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=round(latency_ms, 1),
        ).info("LLM request complete")

        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=response.model,
            latency_ms=latency_ms,
        )
