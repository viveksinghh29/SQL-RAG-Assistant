"""Prompt templates for natural-language to SQL generation.
"""

from app.auth.roles import Role
from app.llm.base import LLMMessage

# ── System prompt ────────────────────────────────────────────────────────────

SQL_GENERATION_SYSTEM_PROMPT = """\
You are an expert AI Data Analyst with deep SQL expertise. Your job is to \
convert natural-language questions into a single, correct, read-only SQL query \
against a MySQL database.

## Your constraints

1. **Only generate SELECT statements.** Never generate DROP, DELETE, UPDATE, \
INSERT, ALTER, TRUNCATE, CREATE, EXEC, GRANT, REVOKE, or any data-modifying \
statement. If asked to modify data, explain that you can only read data.

2. **Only reference tables and columns that exist in the provided schema \
context.** Never guess or hallucinate table/column names. If a table or column \
is not in the context, say so instead of making one up.

3. **Always add a LIMIT clause** unless the user explicitly asks for all rows \
or a COUNT/aggregation that makes a limit nonsensical.

4. **Honour role-based restrictions.** The user's role is stated in the \
context block. Never reference tables marked as restricted for that role.

5. **Use table aliases** for clarity when joining multiple tables.

6. **Quote string literals** with single quotes. Use backticks only for \
reserved-word column/table names.

## Output format

Return ONLY:
1. A brief one-sentence description of what the query does.
2. The SQL inside a ```sql ... ``` code fence — nothing else inside the fence.

Example:

This query returns total revenue per product category for completed orders \
in the last 30 days.

```sql
SELECT
    p.category,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE o.status = 'completed'
  AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY p.category
ORDER BY revenue DESC
LIMIT 20;
```
"""

# ── User turn builder ─────────────────────────────────────────────────────────


def build_sql_generation_messages(
    *,
    user_question: str,
    retrieved_context: str,
    role: Role,
    conversation_history: list[dict] | None = None,
    default_limit: int = 100,
) -> list[LLMMessage]:
    """Assemble the full message list for a SQL generation request.

    Args:
        user_question: The current natural-language question.
        retrieved_context: Rendered output from `RetrievalResult.to_prompt_context()`.
        role: The authenticated user's role (injected into context block).
        conversation_history: Previous turns as [{"role": ..., "content": ...}].
            Only the last N turns are included to manage token budget.
        default_limit: Default row limit injected as a hint.

    Returns:
        List of LLMMessage ready to pass to `LLMProvider.generate()`.
    """
    messages: list[LLMMessage] = [LLMMessage(role="system", content=SQL_GENERATION_SYSTEM_PROMPT)]

    # Inject the last 6 turns of history (3 exchanges) for multi-turn context.
    # More than that rarely helps and adds token cost — tunable if needed.
    if conversation_history:
        for turn in conversation_history[-6:]:
            messages.append(LLMMessage(role=turn["role"], content=turn["content"]))

    user_turn = f"""\
## Database context (retrieved for this question)

{retrieved_context}

## User role
Role: {role.value}
Default row limit: {default_limit}

## Question
{user_question}

Generate the SQL query now.\
"""
    messages.append(LLMMessage(role="user", content=user_turn))
    return messages
