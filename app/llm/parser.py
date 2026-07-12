"""Utilities for parsing structured content out of raw LLM responses.

LLMs occasionally deviate from their output format instructions. Every
parser here is defensive: it tries the happy path first, falls back
gracefully, and raises a typed exception rather than returning None
silently (callers decide how to handle failures).
"""

import json
import re

from app.core.exceptions import LLMResponseParsingError
from app.core.logging import get_logger

log = get_logger("llm.parser")

# Matches ```sql ... ``` or ``` ... ``` (some models omit the language tag)
_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

# Matches a bare SELECT statement in case the model skips the fence entirely
_BARE_SELECT_RE = re.compile(r"(SELECT\s+.+?;)", re.DOTALL | re.IGNORECASE)


def extract_sql(llm_response: str) -> str:
    """Extract the SQL statement from an LLM response.

    Tries, in order:
    1. Content inside a ```sql ... ``` fence (the expected format).
    2. Content inside any ``` ... ``` fence.
    3. A bare SELECT statement in the raw response text.

    Raises `LLMResponseParsingError` if no SQL can be extracted.
    """
    fence_matches = _SQL_FENCE_RE.findall(llm_response)
    if fence_matches:
        # Take the first fence block (there should only be one)
        sql = fence_matches[0].strip()
        if sql:
            return sql

    bare_matches = _BARE_SELECT_RE.findall(llm_response)
    if bare_matches:
        sql = bare_matches[0].strip()
        log.warning("SQL extracted from bare text (no fence found) — LLM deviated from format")
        return sql

    raise LLMResponseParsingError(
        "Could not extract a SQL statement from the LLM response.",
        details={"raw_response_preview": llm_response[:300]},
    )


def extract_json(llm_response: str) -> dict:
    """Extract and parse a JSON object from an LLM response.

    Tries:
    1. Parse the whole response as JSON (ideal case — prompt asked for JSON only).
    2. Extract JSON from inside a ```json ... ``` or ``` ... ``` fence.
    3. Find the first {...} block in the raw text.

    Raises `LLMResponseParsingError` if no valid JSON can be extracted.
    """
    cleaned = llm_response.strip()

    # Happy path: the whole response is valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting from a code fence
    fence_matches = re.findall(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    for block in fence_matches:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # Last resort: find the first balanced {...} block
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    raise LLMResponseParsingError(
        "Could not extract valid JSON from the LLM response.",
        details={"raw_response_preview": llm_response[:300]},
    )


def extract_description(llm_response: str) -> str:
    """Extract the one-sentence query description that precedes the SQL fence.

    The SQL generation prompt asks for: description on first line(s),
    then the ```sql fence. This pulls the description out.
    Returns the full response if no fence is found (fallback).
    """
    fence_start = llm_response.find("```")
    if fence_start == -1:
        return llm_response.strip()
    return llm_response[:fence_start].strip()
