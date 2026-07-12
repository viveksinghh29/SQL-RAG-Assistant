"""SQL RAG Assistant — Streamlit frontend entry point.
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SQL RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from api_client import APIClient
from components import (
    render_assistant_message,
    render_conversation_list,
    render_schema_browser,
    render_user_message,
    thinking_placeholder,
)
from state import State
from styles import inject_styles

inject_styles()
State.init()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_login_page() -> None:
    """Full-page login form shown to unauthenticated users."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🤖 SQL RAG Assistant")
        st.markdown(
            "Ask questions about your data in plain English. "
            "The AI generates SQL, explains results, and visualizes insights."
        )
        st.divider()

        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@company.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign In", use_container_width=True, type="primary"
            )

        if submitted:
            if not email or not password:
                st.error("Please enter your email and password.")
                return

            client = APIClient()
            result = client.login(email, password)

            if result.ok:
                token = result.data["access_token"]
                user = result.data["user"]
                State.set_token(token)
                State.set_user(user)
                st.success(f"Welcome, {user['full_name']}!")
                st.rerun()
            else:
                st.error(result.error.message)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(client: APIClient) -> None:
    """Render the full sidebar: header, new chat, history, schema, logout."""
    user = State.get_user() or {}

    with st.sidebar:
        # ── App header ────────────────────────────────────────────────────
        st.markdown("## 🤖 SQL RAG Assistant")
        role_badge = user.get("role", "").upper()
        st.caption(f"Signed in as **{user.get('full_name', '')}** · `{role_badge}`")
        st.divider()

        # ── New conversation ──────────────────────────────────────────────
        if st.button("➕ New conversation", use_container_width=True, type="primary"):
            State.clear_messages()
            _refresh_conversations(client)
            st.rerun()

        st.markdown("#### 💬 Recent conversations")

        # ── Conversation list ─────────────────────────────────────────────
        conversations = State.get_conversations()
        if not conversations:
            _refresh_conversations(client)
            conversations = State.get_conversations()

        def on_select(conv_id: int) -> None:
            _load_conversation(client, conv_id)
            st.rerun()

        def on_delete(conv_id: int) -> None:
            result = client.delete_conversation(conv_id)
            if result.ok:
                if State.get_conversation_id() == conv_id:
                    State.clear_messages()
                _refresh_conversations(client)
                st.rerun()
            else:
                st.error(result.error.message)

        render_conversation_list(conversations, on_select, on_delete)

        st.divider()

        # ── Schema browser ────────────────────────────────────────────────
        st.markdown("#### 🗂 Database schema")
        schema = State.get_schema()
        if schema is None:
            schema_result = client.get_schema()
            if schema_result.ok:
                State.set_schema(schema_result.data)
                schema = schema_result.data

        if schema:
            render_schema_browser(schema)

        # ── Footer / logout ───────────────────────────────────────────────
        st.divider()
        if st.button("🚪 Sign out", use_container_width=True):
            State.logout()
            st.rerun()


def _refresh_conversations(client: APIClient) -> None:
    result = client.list_conversations()
    if result.ok:
        State.set_conversations(result.data)


def _load_conversation(client: APIClient, conv_id: int) -> None:
    """Load a past conversation into the local message list."""
    result = client.get_conversation(conv_id)
    if not result.ok:
        st.error(result.error.message)
        return

    conv = result.data
    State.clear_messages()
    State.set_conversation_id(conv_id)

    for msg in conv.get("messages", []):
        role = msg.get("role")
        if role == "user":
            State.add_message({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            # Reconstruct a displayable assistant payload from stored data
            State.add_message({
                "role": "assistant",
                "content": msg.get("content", ""),
                "sql": msg.get("generated_sql", ""),
                "status": "success",
                "message_id": msg.get("id"),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def render_chat_page(client: APIClient) -> None:
    """Render the main chat interface with message history and input."""
    st.markdown("### 💬 Ask your data a question")
    st.caption(
        "Try: *'Show total revenue by region last month'* · "
        "*'Which products have the highest return rate?'* · "
        "*'Compare June vs May sales'*"
    )

    # ── Replay stored messages ────────────────────────────────────────────
    for message in State.get_messages():
        if message["role"] == "user":
            render_user_message(message["content"])
        else:
            render_assistant_message(message, client=client)

    # ── Chat input ────────────────────────────────────────────────────────
    question = st.chat_input(
        "Ask a question about your data…",
        disabled=State.is_thinking(),
    )

    if question:
        _handle_user_question(client, question)


def _handle_user_question(client: APIClient, question: str) -> None:
    """Send the user question through the full pipeline and display results."""
    # Display user message immediately
    render_user_message(question)
    State.add_message({"role": "user", "content": question})

    # Show thinking indicator
    placeholder = thinking_placeholder()
    with placeholder:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                '<span class="thinking-text">⏳ Analysing your question…</span>',
                unsafe_allow_html=True,
            )

    State.set_thinking(True)

    try:
        result = client.send_message(
            question=question,
            conversation_id=State.get_conversation_id(),
        )
    finally:
        State.set_thinking(False)
        placeholder.empty()

    if not result.ok:
        with st.chat_message("assistant", avatar="🤖"):
            st.error(f"⚠️ {result.error.message}")
        State.add_message({
            "role": "assistant",
            "content": result.error.message,
            "status": "failed",
            "error_message": result.error.message,
        })
        return

    response = result.data

    # Update active conversation id from the response
    State.set_conversation_id(response.get("conversation_id"))

    # Render and store the assistant response
    render_assistant_message(response, client=client)
    State.add_message({"role": "assistant", **response})

    # Refresh sidebar conversation list (new conversation may have been created)
    _refresh_conversations(client)


# ═══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not State.is_authenticated():
        render_login_page()
        return

    token = State.get_token()
    client = APIClient(token=token)

    # Verify token is still valid (catches expiry between sessions)
    me_result = client.get_me()
    if not me_result.ok:
        State.logout()
        st.warning("Your session has expired. Please sign in again.")
        st.rerun()
        return

    # Keep user profile in sync
    State.set_user(me_result.data)

    render_sidebar(client)
    render_chat_page(client)


if __name__ == "__main__":
    main()
else:
    # Streamlit calls the module at import time; run main() unconditionally
    main()
