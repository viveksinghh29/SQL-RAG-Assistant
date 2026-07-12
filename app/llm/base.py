"""Abstract LLM provider contract.

Every concrete provider (Groq, OpenAI-compatible, ...) implements this
interface. Services depend on `LLMProvider`, never on a concrete SDK
client, so swapping or A/B-testing providers never touches business
logic. Concrete implementations are built in Phase 5 under
`app/llm/providers/`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_ms: float


class LLMProvider(ABC):
    """Contract for a chat-completion-capable LLM provider."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a normalized response.

        Implementations are responsible for: retries on transient errors,
        timeout enforcement, and translating provider-specific exceptions
        into `app.core.exceptions.LLMProviderError`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError
