"""Cached analytics service (Phase 14).

Wraps `AnalyticsService` to cache the two most expensive operations:
  1. LLM explanation generation   (cached by sql + row_count)
  2. Chart type inference         (cached by sql + columns)

Both use short TTLs (same as query results by default) since they depend
on the underlying data being stable. The chart rendering step (pure
Plotly, no LLM) is not cached — it's already fast enough at <10ms.

Savings:
  - Repeat questions in the same conversation: 2 LLM calls → 0
  - Popular queries across users: same SQL → shared cache hits
  - Token costs reduced proportionally to cache hit rate
"""

import json

from app.core.logging import get_logger
from app.llm.base import LLMProvider
from app.query_executor.result import ExecutionResult
from app.utils.cache_client import CacheClient
from app.utils.cache_keys import chart_decision_key, explanation_key
from app.visualization.analytics_service import AnalyticsResult, AnalyticsService
from app.visualization.chart_inference import ChartDecision, ChartInferenceService
from app.visualization.chart_renderer import ChartRenderer
from app.visualization.explanation import ExplanationService

log = get_logger("cache.analytics")


class CachedAnalyticsService:
    """Drop-in replacement for AnalyticsService with Redis caching.

    Implements the same `.analyse()` interface so the API layer swaps
    between cached and uncached by changing one line.
    """

    def __init__(
        self,
        llm: LLMProvider,
        cache: CacheClient,
        renderer: ChartRenderer | None = None,
    ) -> None:
        self._explanation_svc = ExplanationService(llm)
        self._inference_svc = ChartInferenceService(llm)
        self._renderer = renderer or ChartRenderer()
        self._cache = cache

    async def analyse(
        self,
        *,
        question: str,
        execution_result: ExecutionResult,
    ) -> AnalyticsResult:
        """Run cached explanation + chart inference, then render."""
        import asyncio

        rows = execution_result.rows
        columns = execution_result.columns
        sql = execution_result.sql_executed
        row_count = execution_result.row_count

        # Run both cache lookups concurrently
        exp_key = explanation_key(sql, row_count)
        chart_key = chart_decision_key(sql, columns)

        cached_exp, cached_chart = await asyncio.gather(
            self._cache.get(exp_key),
            self._cache.get(chart_key),
        )

        # ── Explanation ───────────────────────────────────────────────────
        if cached_exp is not None:
            explanation = cached_exp
            log.bind(key=exp_key[:20]).debug("Explanation cache HIT")
        else:
            explanation = await self._explanation_svc.explain(
                question=question,
                sql=sql,
                results=rows,
                row_count=row_count,
            )
            await self._cache.set(exp_key, explanation)
            log.bind(key=exp_key[:20]).debug("Explanation cache MISS — cached")

        # ── Chart inference ───────────────────────────────────────────────
        if cached_chart is not None:
            decision = ChartDecision(**cached_chart)
            log.bind(key=chart_key[:20]).debug("Chart inference cache HIT")
        else:
            decision = await self._inference_svc.infer(
                question=question,
                sql=sql,
                columns=columns,
                rows=rows,
            )
            await self._cache.set(
                chart_key,
                {
                    "chart_type": decision.chart_type,
                    "x_column": decision.x_column,
                    "y_column": decision.y_column,
                    "title": decision.title,
                    "rationale": decision.rationale,
                },
            )
            log.bind(key=chart_key[:20]).debug("Chart inference cache MISS — cached")

        # ── Chart rendering (always fresh — fast, no LLM) ─────────────────
        chart_json = self._renderer.render(decision, rows, columns)

        return AnalyticsResult(
            explanation=explanation,
            chart_type=decision.chart_type,
            chart_json=chart_json,
            chart_title=decision.title,
            chart_rationale=decision.rationale,
        )
