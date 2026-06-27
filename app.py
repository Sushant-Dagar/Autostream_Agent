"""
Streamlit chat interface for the AutoStream conversational AI agent.

This file is a thin UI layer only — it does NOT change agent.py / rag.py / tools.py.
It just imports create_agent() and calls .chat() each turn, keeping the
LangGraph AgentState in Streamlit's session_state so the conversation persists
across user turns within a browser session.
"""

import os
import streamlit as st

from agent import create_agent

st.set_page_config(
    page_title="AutoStream Sales Agent",
    page_icon="🎬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar: provider selection + API key entry
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["google", "openai", "anthropic"],
    index=0,
    help="Pick which LLM backend powers the agent. Each needs its own API key below.",
)

PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
env_var = PROVIDER_ENV_VAR[provider]


def get_server_side_key(name: str) -> str:
    """Look for a key configured by the app owner (Streamlit Cloud secrets,
    or a local env var) — never something typed by a visitor."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # st.secrets raises if no secrets.toml exists at all (e.g. local run)
    return os.environ.get(name, "")


server_key = get_server_side_key(env_var)

if server_key:
    # A key is already configured by the app owner — use it silently.
    # NEVER echo it back into a widget; that exposes it in the page's HTML/JS
    # regardless of type="password" masking.
    api_key = server_key
    os.environ[env_var] = server_key
    st.sidebar.success(f"{provider.capitalize()} is ready to go ✅")
else:
    # No server-side key configured — ask the visitor for their own.
    # This box stays EMPTY by default; we never pre-fill it with a real key.
    api_key = st.sidebar.text_input(
        f"{env_var}",
        value="",
        type="password",
        help="Used only for this browser session, never stored or logged.",
    )
    if api_key:
        os.environ[env_var] = api_key

st.sidebar.divider()
if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.pop("agent_state", None)
    st.session_state.pop("chat_history", None)
    st.session_state.pop("agent", None)
    st.rerun()

st.sidebar.caption(
    "This demo wraps the existing LangGraph agent (agent.py) with a chat UI. "
    "Source: github.com/Sushant-Dagar/Autostream_Agent"
)

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
st.title("🎬 AutoStream Sales Agent")
st.caption(
    "A conversational AI agent for **AutoStream**, a SaaS product for automated "
    "video editing. Ask about pricing/features, or say you'd like to sign up "
    "to see the lead-capture flow."
)

if not api_key:
    st.info(f"Enter your **{env_var}** in the sidebar to start chatting.")
    st.stop()

# Lazily build the agent once we have a key, and cache it for this session.
if "agent" not in st.session_state or st.session_state.get("agent_provider") != provider:
    with st.spinner("Setting up the agent..."):
        try:
            st.session_state.agent = create_agent(llm_provider=provider)
            st.session_state.agent_provider = provider
        except Exception as e:
            st.error(f"Couldn't initialize the agent: {e}")
            st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text) for rendering
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None  # LangGraph AgentState, opaque to us

# Render prior turns
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# New user turn
user_input = st.chat_input("Ask about AutoStream, or say you'd like to sign up...")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text, new_state = st.session_state.agent.chat(
                    user_input, st.session_state.agent_state
                )
                st.session_state.agent_state = new_state
            except Exception as e:
                response_text = f"⚠️ Something went wrong calling the LLM: {e}"
        st.markdown(response_text)

    st.session_state.chat_history.append(("assistant", response_text))

# Optional: show captured lead info for demo transparency
lead_info = (st.session_state.agent_state or {}).get("lead_info") if st.session_state.agent_state else None
if lead_info:
    with st.sidebar.expander("📋 Captured lead info (demo)"):
        st.json(lead_info)
