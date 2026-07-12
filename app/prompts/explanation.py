"""Prompt templates for AI result explanation and business insight generation.
"""

import json

from app.llm.base import LLMMessage

# ── System prompt ─────────────────────────────────────────────────────────────

EXPLANATION_SYSTEM_PROMPT = """\
You are a senior business analyst presenting data findings to non-technical \
stakeholders. You receive a SQL query, its results, and the original question. \
Your job is to produce a clear, insightful, business-friendly explanation.

## Your output must include

1. **Summary** (1-2 sentences): What does the data show at a high level?
2. **Key insights** (2-4 bullet points): The most important findings, with \
specific numbers where available.
3. **Recommendation** (1-2 sentences): One concrete, actionable next step \
the business could take based on this data.

## Formatting rules

- Use plain English. Avoid SQL jargon.
- Always reference specific numbers from the results.
- If the result set is empty, explain what that likely means in business terms.
- If the result set is large (>20 rows), focus on top/bottom patterns.
- Keep the entire response under 250 words.
"""

# ── User turn builder ─────────────────────────────────────────────────────────


def build_explanation_messages(
    *,
    original_question: str,
    sql_query: str,
    results: list[dict],
    row_count: int,
) -> list[LLMMessage]:
    """Assemble messages for the explanation/insight generation call.

    Args:
        original_question: The user's original natural-language question.
        sql_query: The SQL that was executed (shown for transparency).
        results: The query results as a list of dicts (up to 20 rows
            included in the prompt — full results are shown in the UI,
            not all passed to the LLM to control token cost).
        row_count: Total rows returned (may exceed len(results) if capped).

    Returns:
        List of LLMMessage ready to pass to `LLMProvider.generate()`.
    """
    # Cap rows sent to LLM at 20 to control token usage. The user sees all
    # rows in the UI table (Phase 12); the LLM only needs enough to generate
    # accurate insights without wasting tokens on row 87.
    sample_rows = results[:20]
    truncated = row_count > 20

    results_block = json.dumps(sample_rows, default=str, indent=2)
    if truncated:
        results_block += f"\n... ({row_count - 20} more rows not shown)"

    user_turn = f"""\
## Original question
{original_question}

## SQL executed
```sql
{sql_query}
```

## Query results ({row_count} row{"s" if row_count != 1 else ""} total)
{results_block}

Please provide your business analysis of these results.\
"""

    return [
        LLMMessage(role="system", content=EXPLANATION_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_turn),
    ]
