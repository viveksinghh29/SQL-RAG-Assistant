"""AI Explanation Service (Phase 9).

Calls the LLM to transform raw SQL query results into a structured
business-language explanation: summary, key insights, and a concrete
recommendation. Uses the prompt template from `app/prompts/explanation.py`
(Phase 5).

Deliberately isolated from the chart generation service — explanation
and chart inference are independent LLM calls with different prompts and
different output shapes. Keeping them separate means either can be
cached, skipped, or retried independently in Phase 14.
"""

import time

from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.prompts.explanation import build_explanation_messages

log = get_logger("analytics.explanation")

# Fallback explanation returned when the LLM call fails — the user still
# sees their data, just without the AI narrative.
_FALLBACK_EXPLANATION = (
    "The query executed successfully. "
    "AI-generated insights are temporarily unavailable — "
    "please review the results directly."
)


class ExplanationService:
    """Generates a natural-language business explanation of query results.

    Args:
        llm: Any `LLMProvider` implementation (Groq, OpenAI, or mock).
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def explain(
        self,
        *,
        question: str,
        sql: str,
        results: list[dict],
        row_count: int,
    ) -> str:
        """Generate a business explanation for the given query results.

        Returns the LLM-generated explanation string, or a fallback
        message if the LLM call fails. Never raises.

        Args:
            question:  Original natural-language question from the user.
            sql:       SQL that was executed (shown for transparency).
            results:   Query result rows as list-of-dicts.
            row_count: Total rows (may exceed len(results) if truncated).
        """
        if not results:
            return self._empty_result_message(question)

        messages = build_explanation_messages(
            original_question=question,
            sql_query=sql,
            results=results,
            row_count=row_count,
        )

        t0 = time.monotonic()
        try:
            response = await self._llm.generate(messages, max_tokens=512)
            latency_ms = (time.monotonic() - t0) * 1000
            log.bind(
                latency_ms=round(latency_ms, 1),
                tokens=response.total_tokens,
            ).info("Explanation generated")
            return response.content.strip()

        except LLMProviderError as exc:
            log.warning(f"Explanation LLM call failed: {exc.message}")
            return _FALLBACK_EXPLANATION

        except Exception as exc:  # noqa: BLE001
            log.exception(f"Unexpected error during explanation generation: {exc}")
            return _FALLBACK_EXPLANATION

    @staticmethod
    def _empty_result_message(question: str) -> str:
        """Return a helpful message when the query returned zero rows."""
        return (
            f"The query for '{question}' returned no results. "
            "This may mean there is no data matching your criteria for the "
            "selected time period, or the filters are too restrictive. "
            "Try broadening your date range or removing some filter conditions."
        )
