"""Unit tests for Phase 9: AI Response Generation & Analytics.

All LLM calls use MockLLMProvider — no network, no API key.
Chart rendering uses real Plotly + pandas against synthetic data.

Coverage:
  ExplanationService:
    - Successful explanation returned from LLM
    - LLM failure → fallback message, no exception
    - Empty results → helpful empty-result message
  ChartInferenceService:
    - Valid JSON response → correct ChartDecision
    - Unsupported chart type → "table" fallback
    - LLM failure → "table" fallback
    - Malformed JSON → "table" fallback
    - Single-column result → "table" fallback (skip inference)
    - Empty result → "table" fallback (skip inference)
  ChartRenderer:
    - bar, line, pie, scatter, area → non-empty JSON
    - chart_type "table" → empty string
    - chart_type "none" → empty string
    - Missing column name → falls back to first column, doesn't crash
    - Empty rows → empty string
  AnalyticsService:
    - Both tasks run and results combined
    - LLM failure in explanation doesn't prevent chart
"""

import json

import pytest

from app.core.exceptions import LLMProviderError
from app.llm.providers.mock_provider import MockLLMProvider
from app.query_executor.result import ExecutionResult, ExecutionStatus
from app.visualization.analytics_service import AnalyticsService
from app.visualization.chart_inference import ChartDecision, ChartInferenceService
from app.visualization.chart_renderer import ChartRenderer
from app.visualization.explanation import ExplanationService

# ── Shared test data ──────────────────────────────────────────────────────────

SAMPLE_ROWS = [
    {"category": "Electronics", "revenue": 45000.0},
    {"category": "Apparel",     "revenue": 32000.0},
    {"category": "Books",       "revenue": 18500.0},
]
SAMPLE_COLUMNS = ["category", "revenue"]

VALID_CHART_JSON = json.dumps({
    "chart_type": "bar",
    "x_column": "category",
    "y_column": "revenue",
    "title": "Revenue by Category",
    "rationale": "Categorical comparison suits a bar chart.",
})


def make_execution_result(rows=None, columns=None, status=ExecutionStatus.SUCCESS):
    rows = rows if rows is not None else SAMPLE_ROWS
    columns = columns if columns is not None else SAMPLE_COLUMNS
    return ExecutionResult(
        status=status,
        rows=rows,
        columns=columns,
        row_count=len(rows),
        sql_executed="SELECT category, SUM(total_amount) AS revenue FROM orders GROUP BY category LIMIT 10;",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ExplanationService
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_explanation_returns_llm_content():
    llm = MockLLMProvider(responses=["Revenue is up 17% driven by Electronics."])
    svc = ExplanationService(llm)
    result = await svc.explain(
        question="Show revenue by category",
        sql="SELECT category, SUM(total_amount) AS revenue FROM orders GROUP BY category",
        results=SAMPLE_ROWS,
        row_count=3,
    )
    assert "17%" in result
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_explanation_llm_failure_returns_fallback():
    llm = MockLLMProvider(raise_on_call=LLMProviderError("quota exceeded"))
    svc = ExplanationService(llm)
    result = await svc.explain(
        question="Show revenue",
        sql="SELECT 1",
        results=SAMPLE_ROWS,
        row_count=3,
    )
    # Must return fallback, never raise
    assert isinstance(result, str)
    assert len(result) > 0
    assert "temporarily unavailable" in result.lower()


@pytest.mark.asyncio
async def test_explanation_empty_results_returns_helpful_message():
    llm = MockLLMProvider()
    svc = ExplanationService(llm)
    result = await svc.explain(
        question="Show June sales",
        sql="SELECT * FROM orders WHERE order_date >= '2024-06-01'",
        results=[],
        row_count=0,
    )
    # Should not call LLM for empty results
    assert llm.call_count == 0
    assert "no results" in result.lower() or "returned no results" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# ChartInferenceService
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_inference_returns_correct_chart_decision():
    llm = MockLLMProvider(responses=[VALID_CHART_JSON])
    svc = ChartInferenceService(llm)
    decision = await svc.infer(
        question="Revenue by category",
        sql="SELECT category, SUM(total_amount) AS revenue FROM orders GROUP BY category",
        columns=SAMPLE_COLUMNS,
        rows=SAMPLE_ROWS,
    )
    assert decision.chart_type == "bar"
    assert decision.x_column == "category"
    assert decision.y_column == "revenue"
    assert decision.title == "Revenue by Category"


@pytest.mark.asyncio
async def test_inference_unsupported_chart_type_falls_back_to_table():
    bad_json = json.dumps({
        "chart_type": "heatmap",  # not in SUPPORTED_CHART_TYPES
        "x_column": "category",
        "y_column": "revenue",
        "title": "Revenue Heatmap",
        "rationale": "Heatmap shows intensity.",
    })
    llm = MockLLMProvider(responses=[bad_json])
    svc = ChartInferenceService(llm)
    decision = await svc.infer(
        question="Show revenue",
        sql="SELECT 1",
        columns=SAMPLE_COLUMNS,
        rows=SAMPLE_ROWS,
    )
    assert decision.chart_type == "table"


@pytest.mark.asyncio
async def test_inference_llm_failure_returns_table_fallback():
    llm = MockLLMProvider(raise_on_call=LLMProviderError("rate limit"))
    svc = ChartInferenceService(llm)
    decision = await svc.infer(
        question="Show revenue",
        sql="SELECT 1",
        columns=SAMPLE_COLUMNS,
        rows=SAMPLE_ROWS,
    )
    assert decision.chart_type == "table"


@pytest.mark.asyncio
async def test_inference_malformed_json_returns_table_fallback():
    llm = MockLLMProvider(responses=["This is not JSON at all."])
    svc = ChartInferenceService(llm)
    decision = await svc.infer(
        question="Show revenue",
        sql="SELECT 1",
        columns=SAMPLE_COLUMNS,
        rows=SAMPLE_ROWS,
    )
    assert decision.chart_type == "table"


@pytest.mark.asyncio
async def test_inference_single_column_skips_llm():
    llm = MockLLMProvider()
    svc = ChartInferenceService(llm)
    decision = await svc.infer(
        question="Count orders",
        sql="SELECT COUNT(*) AS total FROM orders",
        columns=["total"],
        rows=[{"total": 150}],
    )
    assert decision.chart_type == "table"
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_inference_empty_rows_skips_llm():
    llm = MockLLMProvider()
    svc = ChartInferenceService(llm)
    decision = await svc.infer(
        question="Show June sales",
        sql="SELECT * FROM orders WHERE 1=0",
        columns=SAMPLE_COLUMNS,
        rows=[],
    )
    assert decision.chart_type == "table"
    assert llm.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ChartRenderer
# ═══════════════════════════════════════════════════════════════════════════════

def make_decision(chart_type: str) -> ChartDecision:
    return ChartDecision(
        chart_type=chart_type,
        x_column="category",
        y_column="revenue",
        title="Test Chart",
        rationale="Test",
    )


@pytest.mark.parametrize("chart_type", ["bar", "line", "pie", "scatter", "area"])
def test_renderer_produces_valid_plotly_json(chart_type):
    renderer = ChartRenderer()
    chart_json = renderer.render(make_decision(chart_type), SAMPLE_ROWS, SAMPLE_COLUMNS)
    assert chart_json != ""
    parsed = json.loads(chart_json)
    assert "data" in parsed


def test_renderer_table_returns_empty_string():
    renderer = ChartRenderer()
    result = renderer.render(make_decision("table"), SAMPLE_ROWS, SAMPLE_COLUMNS)
    assert result == ""


def test_renderer_none_returns_empty_string():
    renderer = ChartRenderer()
    result = renderer.render(make_decision("none"), SAMPLE_ROWS, SAMPLE_COLUMNS)
    assert result == ""


def test_renderer_empty_rows_returns_empty_string():
    renderer = ChartRenderer()
    result = renderer.render(make_decision("bar"), [], SAMPLE_COLUMNS)
    assert result == ""


def test_renderer_unknown_column_uses_fallback_column():
    """If x_column doesn't exist in data, renderer should use first column."""
    decision = ChartDecision(
        chart_type="bar",
        x_column="nonexistent_col",
        y_column="revenue",
        title="Test",
        rationale="",
    )
    renderer = ChartRenderer()
    # Should not raise — falls back to first column
    result = renderer.render(decision, SAMPLE_ROWS, SAMPLE_COLUMNS)
    assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# AnalyticsService (orchestration)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_analytics_service_returns_combined_result():
    llm = MockLLMProvider(responses=[
        "Revenue grew 17% led by Electronics.",  # explanation
        VALID_CHART_JSON,                        # chart inference
    ])
    svc = AnalyticsService(llm)
    exec_result = make_execution_result()

    result = await svc.analyse(
        question="Revenue by category",
        execution_result=exec_result,
    )

    assert "17%" in result.explanation
    assert result.chart_type == "bar"
    assert result.chart_json != ""
    assert result.chart_title == "Revenue by Category"
    assert llm.call_count == 2  # one for explanation, one for inference


@pytest.mark.asyncio
async def test_analytics_explanation_failure_still_returns_chart():
    """If explanation LLM call fails, chart should still be generated."""
    calls = []

    class PartialFailLLM(MockLLMProvider):
        async def generate(self, messages, **kwargs):
            calls.append(len(messages))
            if len(calls) == 1:
                raise LLMProviderError("quota")
            return await super().generate(messages, **kwargs)

    llm = PartialFailLLM(responses=[VALID_CHART_JSON])
    svc = AnalyticsService(llm)
    exec_result = make_execution_result()

    result = await svc.analyse(
        question="Revenue by category",
        execution_result=exec_result,
    )

    assert "unavailable" in result.explanation.lower()
    assert result.chart_type == "bar"


@pytest.mark.asyncio
async def test_analytics_empty_result_returns_graceful_output():
    llm = MockLLMProvider(responses=[VALID_CHART_JSON])
    svc = AnalyticsService(llm)
    exec_result = make_execution_result(rows=[], columns=[], status=ExecutionStatus.EMPTY)

    result = await svc.analyse(
        question="Show June sales",
        execution_result=exec_result,
    )

    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0
    assert result.chart_json == ""  # no chart for empty data
