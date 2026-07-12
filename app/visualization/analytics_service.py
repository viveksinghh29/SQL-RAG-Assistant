"""Analytics Service (Phase 9).

Orchestrates the two async tasks that run after a successful query
execution:

  1. ExplanationService  — LLM call → business narrative
  2. ChartInferenceService → LLM call → chart decision
     ChartRenderer         → data + decision → Plotly JSON

Tasks 1 and 2 are independent, so they run concurrently via
`asyncio.gather` — the combined latency is max(explanation, inference),
not their sum. Chart rendering (pure CPU, no I/O) runs synchronously
after inference resolves.

Returns an `AnalyticsResult` dataclass consumed by the API layer
(Phase 11) and the Streamlit UI (Phase 12).
"""

import asyncio
from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.query_executor.result import ExecutionResult
from app.visualization.chart_inference import ChartDecision, ChartInferenceService
from app.visualization.chart_renderer import ChartRenderer
from app.visualization.explanation import ExplanationService

log = get_logger("analytics.service")


@dataclass
class AnalyticsResult:
    """Combined output of explanation + chart generation."""
    explanation: str
    chart_type: str          # from ChartDecision (e.g. "bar", "line", "table")
    chart_json: str          # Plotly figure JSON; empty if chart_type is "table"/"none"
    chart_title: str
    chart_rationale: str     # LLM's one-sentence justification (useful for debugging)


class AnalyticsService:
    """Produces AI explanation and chart from an ExecutionResult.

    Args:
        llm:      LLM provider for both explanation and chart inference.
        renderer: ChartRenderer instance (injected for testability).
    """

    def __init__(
        self,
        llm: LLMProvider,
        renderer: ChartRenderer | None = None,
    ) -> None:
        self._explanation_svc = ExplanationService(llm)
        self._inference_svc = ChartInferenceService(llm)
        self._renderer = renderer or ChartRenderer()

    async def analyse(
        self,
        *,
        question: str,
        execution_result: ExecutionResult,
    ) -> AnalyticsResult:
        """Run explanation and chart inference concurrently.

        Args:
            question:         The user's original NL question.
            execution_result: The validated, executed query result from Phase 8.

        Returns:
            AnalyticsResult — always, never raises. Both explanation and
            chart fall back gracefully on LLM failure.
        """
        rows = execution_result.rows
        columns = execution_result.columns
        sql = execution_result.sql_executed

        # Run explanation and chart inference as concurrent tasks
        explanation_task = self._explanation_svc.explain(
            question=question,
            sql=sql,
            results=rows,
            row_count=execution_result.row_count,
        )
        inference_task = self._inference_svc.infer(
            question=question,
            sql=sql,
            columns=columns,
            rows=rows,
        )

        explanation, decision = await asyncio.gather(
            explanation_task,
            inference_task,
            return_exceptions=False,  # both services catch their own errors
        )

        # Chart rendering is synchronous (pure Plotly/pandas, no I/O)
        chart_json = self._renderer.render(decision, rows, columns)

        log.bind(
            chart_type=decision.chart_type,
            has_chart=bool(chart_json),
            explanation_chars=len(explanation),
        ).info("Analytics generation complete")

        return AnalyticsResult(
            explanation=explanation,
            chart_type=decision.chart_type,
            chart_json=chart_json,
            chart_title=decision.title,
            chart_rationale=decision.rationale,
        )
