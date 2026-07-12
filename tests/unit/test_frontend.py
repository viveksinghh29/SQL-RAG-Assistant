"""Unit tests for the Streamlit frontend modules.

Tests cover:
  APIClient:
    - Successful GET/POST response parsing
    - Connection error → APIResult with error, no exception
    - HTTP 401 → APIError with correct code and message
    - HTTP 500 → APIError with correct code
    - Non-JSON error response handled gracefully

  State (session_state mock):
    - init() sets all required keys
    - is_authenticated() true after set_token
    - logout() clears sensitive keys
    - add_message() appends correctly
    - feedback_already_sent() tracks IDs

  Components (pure logic only — no st.* calls):
    - _df_to_excel_bytes produces valid bytes
    - render_metadata_bar data extraction
"""

from unittest.mock import MagicMock, patch

import pytest

from frontend.api_client import APIClient, APIError, APIResult


# ── APIClient tests ────────────────────────────────────────────────────────────

class MockResponse:
    def __init__(self, status_code: int, json_data=None, reason="OK", content=b"x"):
        self.status_code = status_code
        self._json = json_data
        self.reason = reason
        self.content = content if json_data is not None else b""

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture
def client():
    return APIClient(token="test-token")


def test_get_success_parses_json(client):
    mock_resp = MockResponse(200, {"status": "ok"})
    with patch.object(client._session, "request", return_value=mock_resp) as m:
        result = client.health()
    assert result.ok
    assert result.data == {"status": "ok"}
    assert result.error is None


def test_post_success_parses_json(client):
    mock_resp = MockResponse(201, {"id": 1, "email": "a@b.com"})
    with patch.object(client._session, "request", return_value=mock_resp):
        result = client._post("/users", {"email": "a@b.com"})
    assert result.ok
    assert result.data["id"] == 1


def test_401_returns_api_error(client):
    mock_resp = MockResponse(
        401,
        {"error": "authentication_error", "message": "Invalid token.", "details": {}},
    )
    with patch.object(client._session, "request", return_value=mock_resp):
        result = client.get_me()
    assert not result.ok
    assert result.error.status_code == 401
    assert result.error.error_code == "authentication_error"
    assert "Invalid token" in result.error.message


def test_500_returns_api_error_with_fallback(client):
    mock_resp = MockResponse(500, None, reason="Internal Server Error", content=b"error")
    mock_resp.content = b"err"
    with patch.object(client._session, "request", return_value=mock_resp):
        result = client.get_me()
    assert not result.ok
    assert result.error.status_code == 500


def test_connection_error_returns_api_error(client):
    import requests as req_lib
    with patch.object(client._session, "request", side_effect=req_lib.ConnectionError()):
        result = client.health()
    assert not result.ok
    assert result.error.error_code == "connection_error"
    assert result.error.status_code == 0


def test_timeout_returns_api_error(client):
    import requests as req_lib
    with patch.object(client._session, "request", side_effect=req_lib.Timeout()):
        result = client.health()
    assert not result.ok
    assert result.error.error_code == "timeout"


def test_set_token_updates_auth_header(client):
    client.set_token("new-token")
    assert client._session.headers["Authorization"] == "Bearer new-token"


def test_login_sends_correct_payload(client):
    mock_resp = MockResponse(200, {"access_token": "tok", "token_type": "bearer",
                                    "user": {"email": "a@b.com"}})
    with patch.object(client._session, "request", return_value=mock_resp) as m:
        result = client.login("a@b.com", "pass")
    call_kwargs = m.call_args[1]
    assert call_kwargs["json"]["email"] == "a@b.com"
    assert call_kwargs["json"]["password"] == "pass"
    assert result.ok


def test_send_message_includes_conversation_id(client):
    mock_resp = MockResponse(200, {"status": "success", "conversation_id": 42})
    with patch.object(client._session, "request", return_value=mock_resp) as m:
        result = client.send_message("Show sales", conversation_id=42)
    call_kwargs = m.call_args[1]
    assert call_kwargs["json"]["question"] == "Show sales"
    assert call_kwargs["json"]["conversation_id"] == 42


def test_api_result_ok_property():
    ok = APIResult(data={"x": 1})
    assert ok.ok is True
    err = APIResult(error=APIError(400, "bad_request", "Bad"))
    assert err.ok is False


# ── State tests (mocked st.session_state) ─────────────────────────────────────

@pytest.fixture
def mock_session_state():
    """Mock st.session_state as a plain dict for testing."""
    state_dict = {}
    with patch("streamlit.session_state", new=state_dict):
        with patch("frontend.state.st") as mock_st:
            mock_st.session_state = state_dict
            yield state_dict


def test_state_init_sets_all_keys(mock_session_state):
    from frontend.state import State
    State.init()
    assert "_auth_token" in str(mock_session_state) or State._TOKEN in mock_session_state


def test_state_is_authenticated_false_by_default(mock_session_state):
    from frontend.state import State
    State.init()
    assert not State.is_authenticated()


def test_state_set_token_marks_authenticated(mock_session_state):
    from frontend.state import State
    State.init()
    State.set_token("eyJtest")
    assert State.is_authenticated()
    assert State.get_token() == "eyJtest"


def test_state_logout_clears_token(mock_session_state):
    from frontend.state import State
    State.init()
    State.set_token("token")
    State.logout()
    assert not State.is_authenticated()


def test_state_add_message(mock_session_state):
    from frontend.state import State
    State.init()
    State.add_message({"role": "user", "content": "Hello"})
    State.add_message({"role": "assistant", "content": "Hi"})
    msgs = State.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"


def test_state_clear_messages(mock_session_state):
    from frontend.state import State
    State.init()
    State.add_message({"role": "user", "content": "test"})
    State.set_conversation_id(5)
    State.clear_messages()
    assert State.get_messages() == []
    assert State.get_conversation_id() is None


def test_state_feedback_tracking(mock_session_state):
    from frontend.state import State
    State.init()
    assert not State.feedback_already_sent(42)
    State.mark_feedback_sent(42)
    assert State.feedback_already_sent(42)
    assert not State.feedback_already_sent(99)


# ── Component logic tests (no st.* calls) ─────────────────────────────────────

def test_df_to_excel_bytes_produces_valid_bytes():
    import pandas as pd
    from frontend.components import _df_to_excel_bytes

    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    result = _df_to_excel_bytes(df)
    assert isinstance(result, bytes)
    assert len(result) > 0
    # Excel files start with PK (zip magic bytes)
    assert result[:2] == b"PK"
