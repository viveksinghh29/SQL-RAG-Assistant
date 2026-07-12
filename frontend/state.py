"""Streamlit session state management.
"""

from __future__ import annotations
from typing import Any
import streamlit as st


class State:
    """Typed accessors for st.session_state."""

    # Key constants — change the string value once here, everywhere else follows
    _TOKEN = "auth_token"
    _USER = "current_user"
    _CONV_ID = "active_conversation_id"
    _MESSAGES = "chat_messages"
    _CONVERSATIONS = "conversation_list"
    _SCHEMA = "schema_data"
    _THINKING = "is_thinking"
    _FEEDBACK_SENT = "feedback_sent_ids"

    @classmethod
    def init(cls) -> None:
        """Initialise all state keys if they don't exist yet.
        Call once at the top of every Streamlit page."""
        defaults = {
            cls._TOKEN: None,
            cls._USER: None,
            cls._CONV_ID: None,
            cls._MESSAGES: [],
            cls._CONVERSATIONS: [],
            cls._SCHEMA: None,
            cls._THINKING: False,
            cls._FEEDBACK_SENT: set(),
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # ── Auth ──────────────────────────────────────────────────────────────

    @classmethod
    def is_authenticated(cls) -> bool:
        return st.session_state.get(cls._TOKEN) is not None

    @classmethod
    def set_token(cls, token: str) -> None:
        st.session_state[cls._TOKEN] = token

    @classmethod
    def get_token(cls) -> str | None:
        return st.session_state.get(cls._TOKEN)

    @classmethod
    def set_user(cls, user: dict) -> None:
        st.session_state[cls._USER] = user

    @classmethod
    def get_user(cls) -> dict | None:
        return st.session_state.get(cls._USER)

    @classmethod
    def logout(cls) -> None:
        for key in [cls._TOKEN, cls._USER, cls._CONV_ID,
                    cls._MESSAGES, cls._CONVERSATIONS, cls._SCHEMA]:
            st.session_state[key] = None
        st.session_state[cls._MESSAGES] = []
        st.session_state[cls._CONVERSATIONS] = []
        st.session_state[cls._FEEDBACK_SENT] = set()

    # ── Conversation ──────────────────────────────────────────────────────

    @classmethod
    def get_conversation_id(cls) -> int | None:
        return st.session_state.get(cls._CONV_ID)

    @classmethod
    def set_conversation_id(cls, conv_id: int | None) -> None:
        st.session_state[cls._CONV_ID] = conv_id

    @classmethod
    def get_messages(cls) -> list[dict]:
        return st.session_state.get(cls._MESSAGES, [])

    @classmethod
    def add_message(cls, message: dict) -> None:
        if cls._MESSAGES not in st.session_state:
            st.session_state[cls._MESSAGES] = []
        st.session_state[cls._MESSAGES].append(message)

    @classmethod
    def clear_messages(cls) -> None:
        st.session_state[cls._MESSAGES] = []
        st.session_state[cls._CONV_ID] = None

    @classmethod
    def set_conversations(cls, conversations: list[dict]) -> None:
        st.session_state[cls._CONVERSATIONS] = conversations

    @classmethod
    def get_conversations(cls) -> list[dict]:
        return st.session_state.get(cls._CONVERSATIONS, [])

    # ── Schema ────────────────────────────────────────────────────────────

    @classmethod
    def get_schema(cls) -> dict | None:
        return st.session_state.get(cls._SCHEMA)

    @classmethod
    def set_schema(cls, schema: dict) -> None:
        st.session_state[cls._SCHEMA] = schema

    # ── UI state ──────────────────────────────────────────────────────────

    @classmethod
    def set_thinking(cls, thinking: bool) -> None:
        st.session_state[cls._THINKING] = thinking

    @classmethod
    def is_thinking(cls) -> bool:
        return st.session_state.get(cls._THINKING, False)

    @classmethod
    def mark_feedback_sent(cls, message_id: int) -> None:
        st.session_state[cls._FEEDBACK_SENT].add(message_id)

    @classmethod
    def feedback_already_sent(cls, message_id: int) -> bool:
        return message_id in st.session_state.get(cls._FEEDBACK_SENT, set())
