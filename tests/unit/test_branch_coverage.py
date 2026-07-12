"""Branch-coverage gap tests — final push to 90%+ (Phase 15).

Each test targets a specific uncovered branch identified by:
    pytest --cov=app --cov-config=.coveragerc --cov-report=term-missing

Targets:
  login_limiter.py:39       — window eviction branch (old timestamps expelled)
  manager.py:76             — conversation_id provided but conv not found
  manager.py:173-176        — LLM title generation exception fallback
  chunker.py:34,38-41       — overlap carry-forward + hard-split long paragraph
  middleware/rate_limit.py  — 429 response + X-Forwarded-For header branch
  sql_validator/service.py  — column alias detection true branch
  cache_client.py           — set() with empty list sentinel
  llm/parser.py             — bare SELECT fallback warning path
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# login_limiter.py:39 — window eviction
# ═══════════════════════════════════════════════════════════════════════════════

def test_login_limiter_evicts_expired_timestamps():
    """Old timestamps outside the 15-min window must be evicted before checking."""
    import app.auth.login_limiter as ll
    from collections import defaultdict, deque

    EMAIL = "eviction-test@example.com"
    saved = ll._attempts
    fresh = defaultdict(deque)
    ll._attempts = fresh
    try:
        # Manually inject an old timestamp (outside the 900s window)
        fresh[EMAIL].append(time.monotonic() - 1000)  # expired
        fresh[EMAIL].append(time.monotonic() - 1000)  # expired
        fresh[EMAIL].append(time.monotonic() - 1000)  # expired
        fresh[EMAIL].append(time.monotonic() - 1000)  # expired
        fresh[EMAIL].append(time.monotonic() - 1000)  # expired — would trigger if not evicted

        # check_login_allowed should evict all 5 and NOT raise
        ll.check_login_allowed(EMAIL)  # must not raise
    finally:
        ll._attempts = saved


# ═══════════════════════════════════════════════════════════════════════════════
# manager.py:76 — conversation_id not found → creates new
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_manager_creates_new_when_conv_id_not_found():
    """get_or_create_conversation must create new conv when id not found in DB."""
    from app.conversation_memory.manager import ConversationManager
    from app.schemas.chat import ConversationCreate

    # Fake repo that always returns None for get_by_id
    class FakeConvRepo:
        created = []
        _next_id = 1

        async def get_by_id(self, eid):
            return None  # simulates not found

        async def create(self, data, *, user_id=None):
            from dataclasses import dataclass
            @dataclass
            class FakeConv:
                id: int
                user_id: int
                title: str
            conv = FakeConv(id=self._next_id, user_id=user_id or 0, title=data.title)
            self._next_id += 1
            self.created.append(conv)
            return conv

    class FakeMsgRepo:
        async def add_message(self, m): return m

    repo = FakeConvRepo()
    manager = ConversationManager(repo, FakeMsgRepo(), llm=None)

    # Pass a non-None conversation_id that doesn't exist in DB
    conv = await manager.get_or_create_conversation(
        user_id=1, conversation_id=999, first_question="Show me orders"
    )
    assert conv is not None
    assert len(repo.created) == 1  # new conversation was created


# ═══════════════════════════════════════════════════════════════════════════════
# manager.py:173-176 — LLM title generation exception → fallback
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_manager_title_falls_back_on_llm_exception():
    """Title generation must fall back to truncation when LLM raises."""
    from app.conversation_memory.manager import ConversationManager
    from app.llm.providers.mock_provider import MockLLMProvider
    from app.core.exceptions import LLMProviderError

    failing_llm = MockLLMProvider(raise_on_call=Exception("network error"))
    manager = ConversationManager.__new__(ConversationManager)
    manager._llm = failing_llm

    title = await manager._generate_title("Show me revenue by region for last quarter")
    assert isinstance(title, str)
    assert len(title) > 0
    assert len(title) <= 62  # truncated


# ═══════════════════════════════════════════════════════════════════════════════
# chunker.py:34,38-41 — overlap carry-forward + long paragraph hard-split
# ═══════════════════════════════════════════════════════════════════════════════

def test_chunker_overlap_carry_forward():
    """When a new paragraph pushes over max_chars, overlap is carried forward."""
    from app.rag.ingestion.chunker import chunk_text

    # Create text with two paragraphs that together exceed max_chars
    para_a = "A" * 200
    para_b = "B" * 200
    text = f"{para_a}\n\n{para_b}"

    chunks = chunk_text(text, max_chars=250, overlap_chars=50)
    assert len(chunks) >= 2
    # The second chunk should start with overlap from para_a
    assert chunks[1].startswith("A" * 50)


def test_chunker_hard_splits_very_long_paragraph():
    """A single paragraph longer than max_chars must be hard-split."""
    from app.rag.ingestion.chunker import chunk_text

    long_para = "X" * 500
    chunks = chunk_text(long_para, max_chars=200, overlap_chars=20)
    assert len(chunks) >= 3  # 500 chars / 200 max = at least 3 chunks
    for chunk in chunks[:-1]:
        assert len(chunk) <= 200


# ═══════════════════════════════════════════════════════════════════════════════
# middleware/rate_limit.py — 429 response + X-Forwarded-For
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_rate_limiter_returns_429_on_exceeded():
    """RateLimitMiddleware must return 429 when limit is exceeded."""
    from app.middleware.rate_limit import RateLimitMiddleware, _windows
    from collections import deque
    from starlette.testclient import TestClient
    from fastapi import FastAPI

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, requests_per_minute=2)

    @app.get("/api/test")
    async def endpoint():
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)

    # First two requests should pass
    r1 = client.get("/api/test")
    assert r1.status_code == 200
    r2 = client.get("/api/test")
    assert r2.status_code == 200

    # Third request exceeds limit
    r3 = client.get("/api/test")
    assert r3.status_code == 429
    assert "rate_limit_exceeded" in r3.json().get("error", "")

    # Cleanup
    _windows.clear()


def test_rate_limiter_uses_x_forwarded_for():
    """_get_identifier must return X-Forwarded-For IP when header is present."""
    from app.middleware.rate_limit import RateLimitMiddleware

    mock_request = MagicMock()
    mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
    mock_request.client = MagicMock()
    mock_request.client.host = "172.16.0.1"

    identifier = RateLimitMiddleware._get_identifier(mock_request)
    assert identifier == "10.0.0.1"


def test_rate_limiter_falls_back_to_client_host():
    """_get_identifier falls back to client.host when no X-Forwarded-For."""
    from app.middleware.rate_limit import RateLimitMiddleware

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client = MagicMock()
    mock_request.client.host = "192.168.1.50"

    identifier = RateLimitMiddleware._get_identifier(mock_request)
    assert identifier == "192.168.1.50"


def test_rate_limiter_unknown_when_no_client():
    """_get_identifier returns 'unknown' when client is None."""
    from app.middleware.rate_limit import RateLimitMiddleware

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client = None

    identifier = RateLimitMiddleware._get_identifier(mock_request)
    assert identifier == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# sql_validator/service.py — alias detection branch (True path)
# ═══════════════════════════════════════════════════════════════════════════════

def test_validator_is_likely_alias_detects_as_clause():
    """_is_likely_alias_or_expr must return True when col appears as AS alias."""
    from app.sql_validator.service import SQLValidationService

    validator = SQLValidationService()
    sql = "SELECT SUM(total_amount) AS revenue, COUNT(*) AS order_count FROM orders"
    assert validator._is_likely_alias_or_expr(sql, "revenue") is True
    assert validator._is_likely_alias_or_expr(sql, "order_count") is True
    assert validator._is_likely_alias_or_expr(sql, "nonexistent") is False


# ═══════════════════════════════════════════════════════════════════════════════
# cache_client.py — set() with empty list uses sentinel
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_cache_client_set_empty_list_uses_sentinel():
    """set([]) must store _EMPTY_SENTINEL, not '[]'."""
    from app.utils.cache_client import CacheClient, _EMPTY_SENTINEL

    stored = {}

    async def fake_setex(key, ttl, value):
        stored[key] = value

    mock_redis = MagicMock()
    mock_redis.setex = AsyncMock(side_effect=fake_setex)

    client = CacheClient()
    client._client = mock_redis
    await client.set("empty-result", [])

    assert stored.get("empty-result") == _EMPTY_SENTINEL


@pytest.mark.asyncio
async def test_cache_client_set_none_uses_sentinel():
    """set(None) must store _EMPTY_SENTINEL."""
    from app.utils.cache_client import CacheClient, _EMPTY_SENTINEL

    stored = {}

    async def fake_setex(key, ttl, value):
        stored[key] = value

    mock_redis = MagicMock()
    mock_redis.setex = AsyncMock(side_effect=fake_setex)

    client = CacheClient()
    client._client = mock_redis
    await client.set("null-result", None)

    assert stored.get("null-result") == _EMPTY_SENTINEL


# ═══════════════════════════════════════════════════════════════════════════════
# llm/parser.py — bare SELECT fallback (38->41 branch)
# ═══════════════════════════════════════════════════════════════════════════════

def test_parser_bare_select_fallback_executes_warning_branch():
    """The bare SELECT fallback logs a warning — exercises the 38->41 branch."""
    from app.llm.parser import extract_sql

    # No fence — should use bare SELECT regex
    response = "SELECT id, status FROM orders WHERE id = 1 LIMIT 5;"
    sql = extract_sql(response)
    assert "SELECT" in sql.upper()
    assert "orders" in sql.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# query_executor/service.py:126 — _extract_syntax_hint with no 'near' match
# ═══════════════════════════════════════════════════════════════════════════════

def test_extract_syntax_hint_fallback_message():
    """_extract_syntax_hint returns generic message when no 'near' pattern found."""
    from app.query_executor.service import QueryExecutor

    hint = QueryExecutor._extract_syntax_hint("Some generic DB error with no near clause")
    assert "syntax error" in hint.lower()
    assert "rephrase" in hint.lower()


def test_extract_syntax_hint_with_near_pattern():
    """_extract_syntax_hint extracts the near clause from MySQL error."""
    from app.query_executor.service import QueryExecutor

    mysql_error = "You have an error in your SQL syntax; check near 'SELECTT * FROM' at line 1"
    hint = QueryExecutor._extract_syntax_hint(mysql_error)
    assert "SELECTT" in hint
    assert "near" in hint.lower()
