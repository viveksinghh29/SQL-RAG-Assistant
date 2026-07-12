"""Typed HTTP client for centralized, synchronous communication between the Streamlit frontend and SQL RAG Assistant backend API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests


# Base URL — reads from env so Docker Compose can override it
_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
_TIMEOUT = 60  # seconds — generous for slow LLM calls


@dataclass
class APIError:
    status_code: int
    error_code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class APIResult:
    """Thin wrapper: either `data` (success) or `error` (failure)."""
    data: Any = None
    error: APIError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class APIClient:
    """HTTP client for the SQL RAG backend. One instance per Streamlit session."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    def set_token(self, token: str) -> None:
        self._token = token
        self._session.headers["Authorization"] = f"Bearer {token}"

    # ── Auth ──────────────────────────────────────────────────────────────

    def login(self, email: str, password: str) -> APIResult:
        return self._post("/auth/login", {"email": email, "password": password})

    def get_me(self) -> APIResult:
        return self._get("/auth/me")

    # ── Chat ──────────────────────────────────────────────────────────────

    def send_message(self, question: str, conversation_id: int | None) -> APIResult:
        return self._post("/chat", {
            "question": question,
            "conversation_id": conversation_id,
        })

    # ── History ───────────────────────────────────────────────────────────

    def list_conversations(self, limit: int = 50) -> APIResult:
        return self._get(f"/conversations?limit={limit}")

    def get_conversation(self, conversation_id: int) -> APIResult:
        return self._get(f"/conversations/{conversation_id}")

    def delete_conversation(self, conversation_id: int) -> APIResult:
        return self._delete(f"/conversations/{conversation_id}")

    # ── Schema ────────────────────────────────────────────────────────────

    def get_schema(self) -> APIResult:
        return self._get("/schema")

    # ── Feedback ──────────────────────────────────────────────────────────

    def submit_feedback(
        self, message_id: int, is_positive: bool, comment: str | None = None
    ) -> APIResult:
        return self._post("/feedback", {
            "message_id": message_id,
            "is_positive": is_positive,
            "comment": comment,
        })

    # ── Health ────────────────────────────────────────────────────────────

    def health(self) -> APIResult:
        return self._get("/health")

    # ── Private HTTP helpers ──────────────────────────────────────────────

    def _get(self, path: str) -> APIResult:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> APIResult:
        return self._request("POST", path, json=body)

    def _delete(self, path: str) -> APIResult:
        return self._request("DELETE", path)

    def _request(self, method: str, path: str, **kwargs) -> APIResult:
        url = f"{_BASE_URL}{path}"
        try:
            resp = self._session.request(method, url, timeout=_TIMEOUT, **kwargs)
        except requests.ConnectionError:
            return APIResult(error=APIError(
                status_code=0,
                error_code="connection_error",
                message="Cannot reach the API server. Is the backend running?",
            ))
        except requests.Timeout:
            return APIResult(error=APIError(
                status_code=0,
                error_code="timeout",
                message="The request timed out. The query may still be processing.",
            ))

        if resp.status_code in (200, 201):
            if resp.status_code == 204 or not resp.content:
                return APIResult(data={})
            return APIResult(data=resp.json())

        # Structured error from our API
        try:
            body = resp.json()
            return APIResult(error=APIError(
                status_code=resp.status_code,
                error_code=body.get("error", "unknown_error"),
                message=body.get("message", "An error occurred."),
                details=body.get("details", {}),
            ))
        except Exception:
            return APIResult(error=APIError(
                status_code=resp.status_code,
                error_code="http_error",
                message=f"HTTP {resp.status_code}: {resp.reason}",
            ))
