"""Chart type inference service (Phase 9).

Asks the LLM to decide the best chart type for a given result set using
the structured JSON prompt from `app/prompts/visualization.py` (Phase 5).
The decision is validated against a known set of supported chart types
before being returned — if the LLM hallucinates an unknown type, we fall
back to "table" rather than crashing the rendering step.

Inference is intentionally a separate service from rendering (ChartRenderer)
so:
  - The inference result can be cached in Phase 14 (same query → same chart)
  - The renderer can be tested without LLM calls
  - Future versions can skip the LLM and use a heuristic engine instead
"""

from dataclasses import dataclass

from app.core.exceptions import LLMProviderError, LLMResponseParsingError
from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.llm.parser import extract_json
from app.prompts.visualization import build_viz_inference_messages

log = get_logger("analytics.chart_inference")

# Canonical set of chart types this application can render.
# Must stay in sync with ChartRenderer.RENDERERS in chart_renderer.py.
SUPPORTED_CHART_TYPES: frozenset[str] = frozenset({
    "bar", "line", "pie", "scatter", "area", "table", "none",
})


@dataclass(frozen=True, slots=True)
class ChartDecision:
    """Structured chart-type decision from the LLM."""
    chart_type: str      # one of SUPPORTED_CHART_TYPES
    x_column: str        # column name for x-axis / labels
    y_column: str        # column name for y-axis / values
    title: str           # descriptive chart title
    rationale: str       # one-sentence justification


class ChartInferenceService:
    """Asks the LLM to decide the best chart type for a result set.

    Args:
        llm: Any `LLMProvider` implementation.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def infer(
        self,
        *,
        question: str,
        sql: str,
        columns: list[str],
        rows: list[dict],
    ) -> ChartDecision:
        """Return the best chart decision for this result set.

        Falls back to `ChartDecision(chart_type="table", ...)` if the
        LLM call fails or returns an unsupported type. Never raises.

        Args:
            question: Original NL question (gives the LLM intent context).
            sql:      Executed SQL (shows grouping/aggregation structure).
            columns:  Column names from the result set.
            rows:     Up to 5 sample rows for the LLM to inspect.
        """
        # Skip inference for edge cases where the answer is obvious
        if not rows:
            return self._fallback("table", "No data to visualize", question)
        if len(columns) == 1:
            return self._fallback("table", "Single-column result", question)

        messages = build_viz_inference_messages(
            original_question=question,
            sql_query=sql,
            columns=columns,
            sample_rows=rows[:5],
        )

        try:
            response = await self._llm.generate(messages, max_tokens=256)
        except LLMProviderError as exc:
            log.warning(f"Chart inference LLM call failed: {exc.message}")
            return self._fallback("table", "LLM unavailable", question)

        try:
            data = extract_json(response.content)
        except LLMResponseParsingError as exc:
            log.warning(f"Chart inference JSON parse failed: {exc.message}")
            return self._fallback("table", "Parse error", question)

        chart_type = str(data.get("chart_type", "table")).lower().strip()
        if chart_type not in SUPPORTED_CHART_TYPES:
            log.warning(
                f"LLM returned unsupported chart type '{chart_type}', "
                "defaulting to 'table'"
            )
            chart_type = "table"

        decision = ChartDecision(
            chart_type=chart_type,
            x_column=str(data.get("x_column", columns[0] if columns else "")),
            y_column=str(data.get("y_column", columns[-1] if len(columns) > 1 else "")),
            title=str(data.get("title", question[:60])),
            rationale=str(data.get("rationale", "")),
        )

        log.bind(
            chart_type=decision.chart_type,
            x=decision.x_column,
            y=decision.y_column,
        ).info("Chart type inferred")

        return decision

    @staticmethod
    def _fallback(chart_type: str, reason: str, question: str) -> ChartDecision:
        return ChartDecision(
            chart_type=chart_type,
            x_column="",
            y_column="",
            title=question[:60],
            rationale=reason,
        )
