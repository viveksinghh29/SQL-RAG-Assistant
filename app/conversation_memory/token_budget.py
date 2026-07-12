"""Token budget estimation for context window management.
"""

from dataclasses import dataclass


# Groq / Llama 3 70B context window.  We reserve a conservative portion
# for the system prompt, retrieved RAG context, and the new user turn,
# leaving this budget for conversation history.
LLAMA3_CONTEXT_TOKENS = 8_192
_HISTORY_BUDGET_FRACTION = 0.30   # use at most 30 % of context for history
HISTORY_TOKEN_BUDGET = int(LLAMA3_CONTEXT_TOKENS * _HISTORY_BUDGET_FRACTION)

# Approximate characters per token for English + SQL text
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    tokens: int
    chars: int

    @property
    def within_budget(self) -> bool:
        return self.tokens <= HISTORY_TOKEN_BUDGET


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in `text` using the 4-chars heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict]) -> TokenEstimate:
    """Estimate total tokens for a list of {role, content} message dicts."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    # Add ~4 tokens overhead per message for role + formatting
    overhead = len(messages) * 4
    total_tokens = (total_chars // _CHARS_PER_TOKEN) + overhead
    return TokenEstimate(tokens=total_tokens, chars=total_chars)


def trim_to_budget(
    messages: list[dict],
    *,
    budget: int = HISTORY_TOKEN_BUDGET,
) -> list[dict]:
    """Trim a message list from the oldest end to fit within `budget` tokens.

    Always preserves the most recent messages. Removes whole turn pairs
    (user + assistant) rather than splitting a pair.
    Returns an empty list if even a single message exceeds the budget.
    """
    if not messages:
        return []

    # Work backwards from the most recent message, accumulating tokens
    kept: list[dict] = []
    token_count = 0

    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg.get("content", "")) + 4
        if token_count + msg_tokens > budget:
            break
        kept.append(msg)
        token_count += msg_tokens

    return list(reversed(kept))
