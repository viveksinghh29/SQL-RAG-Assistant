"""Mock LLM provider for testing.

Usage in tests:

    from app.llm.providers.mock_provider import MockLLMProvider

    provider = MockLLMProvider(
        responses=["SELECT 1;"],  # consumed in order, last one repeats
    )
    response = await provider.generate(messages)
    assert response.content == "SELECT 1;"

Setting `raise_on_call` simulates provider failures:

    provider = MockLLMProvider(raise_on_call=LLMProviderError("quota"))
"""

from app.core.exceptions import LLMProviderError
from app.llm.base import LLMMessage, LLMProvider, LLMResponse

_DEFAULT_SQL_RESPONSE = """\
This query returns all completed orders in the last 30 days.

```sql
SELECT id, customer_id, order_date, total_amount
FROM orders
WHERE status = 'completed'
  AND order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
ORDER BY order_date DESC
LIMIT 100;
```
"""


class MockLLMProvider(LLMProvider):
    """Fully controllable mock for use in tests — no network, no API key."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        raise_on_call: Exception | None = None,
        model: str = "mock-model",
    ) -> None:
        self._responses = responses or [_DEFAULT_SQL_RESPONSE]
        self._raise = raise_on_call
        self._model = model
        self.call_count = 0
        self.last_messages: list[LLMMessage] | None = None

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
        self.call_count += 1
        self.last_messages = messages

        if self._raise is not None:
            raise self._raise

        # Consume responses in order; repeat the last one when exhausted
        idx = min(self.call_count - 1, len(self._responses) - 1)
        content = self._responses[idx]

        return LLMResponse(
            content=content,
            prompt_tokens=len(" ".join(m.content for m in messages)) // 4,
            completion_tokens=len(content) // 4,
            total_tokens=(len(" ".join(m.content for m in messages)) + len(content)) // 4,
            model=self._model,
            latency_ms=1.0,
        )
