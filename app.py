"""Stage 1 Streamlit placeholder for the AI Task Manager."""

import streamlit as st

from task_manager.config import get_settings


st.set_page_config(page_title="AI Task Manager", page_icon="✅", layout="wide")

settings = get_settings()

st.title("AI Task Manager")
st.caption("Stage 1 · Project foundation")
st.info("The project scaffold is ready. AI chat and task operations arrive in later stages.")

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Provider:** {settings.llm_provider}")
    st.write(f"**Model:** {settings.llm_model}")
    st.write(f"**Timezone:** {settings.app_timezone}")
    st.caption("API credentials are never displayed.")

with st.chat_message("assistant"):
    st.write(
        "Welcome! The chat interface is a placeholder for now. "
        "No database, vector search or AI actions are active yet."
    )

st.chat_input("Chat will be enabled in a later stage", disabled=True)
