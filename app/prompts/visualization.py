"""Prompt template for automatic chart-type inference.
"""

import json

from app.llm.base import LLMMessage

VIZ_SYSTEM_PROMPT = """\
You are a data visualization expert. Given a SQL query, its results, and the \
original question, decide the single best chart type to represent the data.

## Supported chart types and when to use them
- bar: comparing categorical values (e.g. revenue by category)
- line: trends over time (time-series with continuous x-axis)
- pie: part-of-whole breakdown (proportions that sum to 100%)
- scatter: correlation between two numeric variables
- area: cumulative trend over time
- table: raw data with many columns, no obvious visual pattern
- none: aggregations returning a single number (e.g. total revenue = $X)

## Output format — JSON only, no prose

{
  "chart_type": "<one of the types above>",
  "x_column": "<column name for x-axis or labels>",
  "y_column": "<column name for y-axis or values>",
  "title": "<short descriptive chart title>",
  "rationale": "<one sentence explaining the choice>"
}

Return ONLY the JSON object. No markdown fences, no explanation outside the JSON.
"""


def build_viz_inference_messages(
    *,
    original_question: str,
    sql_query: str,
    columns: list[str],
    sample_rows: list[dict],
) -> list[LLMMessage]:
    """Assemble messages to ask the LLM which chart type fits these results."""
    preview = json.dumps(sample_rows[:5], default=str, indent=2)

    user_turn = f"""\
## Original question
{original_question}

## SQL query
```sql
{sql_query}
```

## Result columns
{columns}

## Sample rows (up to 5)
{preview}

Choose the best chart type for this data.\
"""
    return [
        LLMMessage(role="system", content=VIZ_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_turn),
    ]
