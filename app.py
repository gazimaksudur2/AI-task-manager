"""Stage 4: Single-page Streamlit interface for AI Task Manager (redesigned UI)."""

import html
import logging
from datetime import date, datetime
from uuid import uuid4

import streamlit as st

from task_manager.config import get_settings
from task_manager.database import TaskDatabase
from task_manager.graph import build_agent_graph
from task_manager.task_service import TaskService
from task_manager.vector_store import TaskVectorStore


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Task Manager",
    page_icon="✅",
    layout="wide",
)

WELCOME = (
    "Hello! Tell me what to do with your tasks. "
    "You can also ask me to list or search them."
)

# Starter prompts shown on an empty conversation. They go through the
# same LangGraph path as anything typed by the user.
STARTER_PROMPTS = [
    "Show all my tasks",
    "What is still in progress?",
    "Show high priority tasks that are not done",
]

CHAT_HEIGHT = 540


# --------------------------------------------------
# 0. THEME (CSS)
# --------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

:root {
  --bg: #EEF1F6;
  --surface: #FFFFFF;
  --ink: #0F1B33;
  --muted: #5B6780;
  --line: #D9DFEA;
  --accent: #2B4DFF;
  --accent-soft: #E4E9FF;
  --todo: #5B6780;       --todo-bg: #E8ECF3;
  --prog: #A8530A;       --prog-bg: #FDEBCB;
  --done: #0F7B53;       --done-bg: #D9F2E6;
  --p-low: #9AA7BD;      --p-med: #E0A100;   --p-high: #D6334A;
}

.stApp {
  background: var(--bg);
  color: var(--ink);
  font-family: 'Figtree', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu { visibility: hidden; }
.block-container { max-width: 1280px; padding-top: 1.6rem; padding-bottom: 2rem; }

/* ---------- Masthead ---------- */
.tm-hero {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1.5rem; flex-wrap: wrap;
  padding: 0.4rem 0 1.2rem 0; margin-bottom: 1.2rem;
  border-bottom: 2px solid var(--ink);
}
.tm-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800; font-size: clamp(2rem, 4vw, 3rem);
  line-height: 1; letter-spacing: -0.03em; margin: 0;
}
.tm-title .tick {
  display: inline-grid; place-items: center;
  width: 0.9em; height: 0.9em; margin-right: 0.28em;
  background: var(--accent); color: #fff; border-radius: 0.24em;
  font-size: 0.78em; vertical-align: 0.04em;
}
.tm-sub { margin: 0.6rem 0 0 0; color: var(--muted); font-size: 1.02rem; max-width: 46ch; }
.tm-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.tm-pill {
  display: inline-flex; align-items: center; gap: 0.45rem;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 999px; padding: 0.32rem 0.8rem;
  font-size: 0.85rem; font-weight: 600; color: var(--ink);
}
.tm-pill i { width: 8px; height: 8px; border-radius: 50%; background: var(--done); display: inline-block; }
.tm-pill.warn i { background: var(--p-med); }
.tm-pill.bad i { background: var(--p-high); }

/* ---------- Section headings ---------- */
.tm-h {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700; font-size: 1.45rem; letter-spacing: -0.02em;
  margin: 0; padding: 0.15rem 0 0.5rem 0;
}
.tm-h small {
  font-family: 'Figtree', sans-serif; font-weight: 600;
  font-size: 0.82rem; color: var(--muted); margin-left: 0.5rem; letter-spacing: 0;
}

/* ---------- Stats ---------- */
.tm-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; margin-bottom: 0.6rem; }
.tm-stat {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 0.6rem 0.8rem;
}
.tm-stat b {
  display: block; font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.7rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;
}
.tm-stat span { color: var(--muted); font-size: 0.82rem; font-weight: 600; }
.tm-stat.todo b { color: var(--todo); }
.tm-stat.prog b { color: var(--prog); }
.tm-stat.done b { color: var(--done); }
.tm-bar {
  display: flex; height: 8px; border-radius: 99px; overflow: hidden;
  background: var(--line); margin-bottom: 1rem;
}
.tm-bar span { display: block; height: 100%; }
.tm-bar .b-done { background: var(--done); }
.tm-bar .b-prog { background: #E39A2D; }
.tm-bar .b-todo { background: #AEB8CB; }

/* ---------- Task cards ---------- */
.tm-card {
  position: relative; background: var(--surface);
  border: 1px solid var(--line); border-radius: 12px;
  padding: 0.8rem 1rem 0.8rem 1.2rem; margin-bottom: 0.15rem;
  overflow: hidden;
}
.tm-card::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
  background: var(--p-low);
}
.tm-card.p-medium::before { background: var(--p-med); }
.tm-card.p-high::before { background: var(--p-high); }
.tm-card.is-done .tm-task-title { color: var(--muted); text-decoration: line-through; text-decoration-thickness: 1px; }
.tm-top { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
.tm-id {
  font-size: 0.78rem; font-weight: 700; color: var(--accent);
  background: var(--accent-soft); border-radius: 6px; padding: 0.1rem 0.45rem;
  font-variant-numeric: tabular-nums;
}
.tm-task-title { font-size: 1.04rem; font-weight: 600; margin: 0.45rem 0 0.55rem 0; line-height: 1.35; }
.tm-meta { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
.tm-tag {
  font-size: 0.78rem; font-weight: 700; border-radius: 6px; padding: 0.15rem 0.55rem;
}
.tm-tag.s-todo { background: var(--todo-bg); color: var(--todo); }
.tm-tag.s-in-progress { background: var(--prog-bg); color: var(--prog); }
.tm-tag.s-done { background: var(--done-bg); color: var(--done); }
.tm-tag.pr { background: transparent; border: 1px solid var(--line); color: var(--muted); }
.tm-due { font-size: 0.82rem; color: var(--muted); font-weight: 600; margin-left: auto; }
.tm-due.late { color: var(--p-high); }
.tm-empty {
  border: 2px dashed var(--line); border-radius: 12px; padding: 1.6rem 1rem;
  text-align: center; color: var(--muted); background: rgba(255,255,255,0.5);
}
.tm-empty b { color: var(--ink); display: block; margin-bottom: 0.2rem; font-size: 1.02rem; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 14px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--accent-soft); border-color: #C9D3FF;
}
[data-testid="stChatInput"] { border-radius: 14px; border: 1.5px solid var(--ink); background: var(--surface); }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 3px var(--accent-soft); border-color: var(--accent); }

/* ---------- Controls ---------- */
.stButton > button {
  border-radius: 10px; border: 1.5px solid var(--ink); background: var(--surface);
  color: var(--ink); font-weight: 600; transition: background .12s, color .12s;
}
.stButton > button:hover { background: var(--ink); color: #fff; border-color: var(--ink); }
.stButton > button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
[data-baseweb="select"] > div { border-radius: 10px; background: var(--surface); }
[data-testid="stExpander"] { border-radius: 10px; background: var(--surface); }
[data-testid="stAlert"] { border-radius: 12px; }

@media (max-width: 760px) {
  .tm-stats { grid-template-columns: repeat(2, 1fr); }
  .tm-due { margin-left: 0; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 1. SESSION STATE
# --------------------------------------------------

def initialize_chat_state() -> None:
    """Initialize state only once per browser session."""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME,
                "kind": "info",
            }
        ]

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())

    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False


def queue_prompt(text: str) -> None:
    """Button callback: hand a starter prompt to the chat flow."""
    st.session_state.queued_prompt = text


# --------------------------------------------------
# 2. INITIALIZE THE EXISTING BACKEND
# --------------------------------------------------

def initialize_runtime(settings) -> None:
    """Reuse services and graph across Streamlit reruns."""

    if "database" not in st.session_state:
        st.session_state.database = TaskDatabase(
            settings.sqlite_path
        )

    if (
        "service" not in st.session_state
        and "index_error" not in st.session_state
    ):
        try:
            with st.spinner("Opening semantic task index..."):
                index = TaskVectorStore(
                    settings.chroma_path
                )

                service = TaskService(
                    st.session_state.database,
                    index,
                )

                # Repair stale or missing Chroma records
                # using authoritative SQLite records.
                service.reconcile_index()

                st.session_state.service = service

        except Exception:
            logger.exception(
                "Could not initialize the task search index"
            )
            st.session_state.index_error = True

    if (
        st.session_state.get("service") is not None
        and settings.provider_key_configured
        and "graph" not in st.session_state
        and "graph_error" not in st.session_state
    ):
        try:
            st.session_state.graph = build_agent_graph(
                st.session_state.service
            )

        except Exception:
            logger.exception(
                "Could not initialize the AI graph"
            )
            st.session_state.graph_error = True


# --------------------------------------------------
# 3. MASTHEAD
# --------------------------------------------------

def render_masthead(settings) -> None:
    """Title plus live status of the AI workflow and search index."""

    if not settings.provider_key_configured:
        ai_pill = '<span class="tm-pill warn"><i></i>AI chat off</span>'
    elif st.session_state.get("graph_error"):
        ai_pill = '<span class="tm-pill bad"><i></i>AI workflow failed</span>'
    else:
        label = html.escape(
            f"{settings.llm_provider} · {settings.llm_model}"
        )
        ai_pill = f'<span class="tm-pill"><i></i>{label}</span>'

    if st.session_state.get("index_error"):
        index_pill = (
            '<span class="tm-pill bad"><i></i>Search index unavailable</span>'
        )
    elif st.session_state.get("service") is not None:
        index_pill = (
            '<span class="tm-pill"><i></i>Semantic search ready</span>'
        )
    else:
        index_pill = (
            '<span class="tm-pill warn"><i></i>Search index starting</span>'
        )

    st.markdown(
        '<div class="tm-hero">'
        "<div>"
        '<div class="tm-title"><span class="tick">✓</span>AI Task Manager</div>'
        '<p class="tm-sub">Manage tasks in plain English. '
        "SQLite keeps your tasks; ChromaDB powers semantic search.</p>"
        "</div>"
        f'<div class="tm-pills">{ai_pill}{index_pill}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 4. RENDER CHAT MESSAGES
# --------------------------------------------------

def render_message(message: dict) -> None:
    """Display chat messages with appropriate feedback."""

    avatar = "✅" if message["role"] == "assistant" else "🧑"

    with st.chat_message(message["role"], avatar=avatar):
        kind = message.get("kind")

        if kind == "clarification":
            st.warning(message["content"])

        elif kind == "error":
            st.error(message["content"])

        else:
            st.markdown(message["content"])


# --------------------------------------------------
# 5. HANDLE A USER INSTRUCTION
# --------------------------------------------------

def handle_prompt(prompt: str, graph) -> None:
    """Send instructions to LangGraph, never directly to SQL."""

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    try:
        result = graph.invoke(
            {"user_request": prompt},
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            },
        )

        response = result.get("response") or (
            "The agent returned no confirmation. "
            "Review your task list before retrying."
        )

        pending = bool(
            result.get("pending_clarification")
        )

        execution_results = (
            result.get("execution_results") or []
        )

        failed = (
            bool(result.get("validation_errors"))
            or any(
                entry.get("kind") == "read_error"
                for entry in execution_results
            )
        )

        if pending:
            kind = "clarification"
        elif failed:
            kind = "error"
        else:
            kind = "normal"

        st.session_state.awaiting_clarification = pending

    except Exception:
        logger.exception(
            "LangGraph invocation failed"
        )

        response = (
            "The request could not be confirmed. "
            "Check the SQLite task list before retrying; "
            "a database change might already have committed."
        )

        kind = "error"

        # Preserve the existing clarification state
        # if an unexpected graph error occurs.

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "kind": kind,
        }
    )


# --------------------------------------------------
# 6. TASK LIST
# --------------------------------------------------

def _slug(value: str) -> str:
    return str(value).strip().lower().replace(" ", "-")


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def render_stats(reader) -> None:
    """Overview of all saved tasks, ignoring the list filters."""

    try:
        everything = reader.list_tasks(status=None, priority=None)
    except Exception:
        logger.exception("Could not compute task overview")
        return

    total = len(everything)
    counts = {"todo": 0, "in-progress": 0, "done": 0}
    for task in everything:
        key = _slug(task.status.value)
        if key in counts:
            counts[key] += 1

    st.markdown(
        '<div class="tm-stats">'
        f'<div class="tm-stat"><b>{total}</b><span>Total</span></div>'
        f'<div class="tm-stat todo"><b>{counts["todo"]}</b><span>Todo</span></div>'
        f'<div class="tm-stat prog"><b>{counts["in-progress"]}</b><span>In progress</span></div>'
        f'<div class="tm-stat done"><b>{counts["done"]}</b><span>Done</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if total:
        pct = {k: v * 100 / total for k, v in counts.items()}
        st.markdown(
            '<div class="tm-bar" role="img" '
            f'aria-label="{counts["done"]} of {total} tasks done">'
            f'<span class="b-done" style="width:{pct["done"]:.1f}%"></span>'
            f'<span class="b-prog" style="width:{pct["in-progress"]:.1f}%"></span>'
            f'<span class="b-todo" style="width:{pct["todo"]:.1f}%"></span>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_task_card(task) -> None:
    status = task.status.value
    priority = task.priority.value
    due_date = _as_date(task.due_date)

    late = (
        due_date is not None
        and due_date < date.today()
        and _slug(status) != "done"
    )

    if due_date:
        due_text = f"Due {due_date.isoformat()}" + (" · overdue" if late else "")
    else:
        due_text = "No due date"

    classes = f"tm-card p-{_slug(priority)}"
    if _slug(status) == "done":
        classes += " is-done"

    st.markdown(
        f'<div class="{classes}">'
        '<div class="tm-top">'
        f'<span class="tm-id">{html.escape(str(task.id))}</span>'
        "</div>"
        f'<div class="tm-task-title">{html.escape(str(task.title))}</div>'
        '<div class="tm-meta">'
        f'<span class="tm-tag s-{_slug(status)}">{html.escape(status)}</span>'
        f'<span class="tm-tag pr">{html.escape(priority)} priority</span>'
        f'<span class="tm-due{" late" if late else ""}">{html.escape(due_text)}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if task.description:
        with st.expander("Description"):
            st.write(task.description)


def show_task_list(service, database) -> None:
    """Read authoritative task records with optional filters."""

    heading_col, refresh_col = st.columns([3, 1])

    with heading_col:
        st.markdown('<div class="tm-h">Tasks</div>', unsafe_allow_html=True)

    with refresh_col:
        if st.button(
            "↻ Refresh",
            key="refresh_tasks",
            use_container_width=True,
        ):
            st.rerun()

    # Normally use TaskService.
    # If ChromaDB is unavailable, allow read-only
    # access to the authoritative SQLite database.
    reader = service if service is not None else database

    render_stats(reader)

    status_col, priority_col = st.columns(2)

    with status_col:
        status = st.selectbox(
            "Status",
            ["All", "Todo", "In progress", "Done"],
            key="status_filter",
        )

    with priority_col:
        priority = st.selectbox(
            "Priority",
            ["All", "Low", "Medium", "High"],
            key="priority_filter",
        )

    filters = {
        "status": None if status == "All" else status,
        "priority": None if priority == "All" else priority,
    }

    try:
        tasks = reader.list_tasks(**filters)
        total = reader.count_tasks()

    except Exception:
        logger.exception(
            "Could not read tasks"
        )

        st.error(
            "Could not load tasks from SQLite. "
            "Check local database access."
        )
        return

    st.caption(
        f"{len(tasks)} shown · {total} total saved"
    )

    with st.container(height=CHAT_HEIGHT - 130, border=False):
        if not tasks:
            if total == 0:
                st.markdown(
                    '<div class="tm-empty"><b>No tasks yet</b>'
                    "Ask the assistant to add one, for example "
                    "&ldquo;Add a high priority task to renew my passport.&rdquo;"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="tm-empty"><b>No tasks match these filters</b>'
                    "Set Status and Priority back to All to see everything."
                    "</div>",
                    unsafe_allow_html=True,
                )
            return

        for task in tasks:
            render_task_card(task)


# --------------------------------------------------
# 7. MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def main() -> None:
    inject_css()
    initialize_chat_state()

    try:
        settings = get_settings()
        initialize_runtime(settings)

    except Exception:
        logger.exception(
            "Unable to initialize configuration or local task storage"
        )

        st.error(
            "The app could not open its local task database "
            "or configuration. Check your .env and data "
            "directory, then restart Streamlit."
        )

        st.stop()

    service = st.session_state.get("service")
    graph = st.session_state.get("graph")
    database = st.session_state.database

    render_masthead(settings)

    # ----------------------------------------------
    # CONFIGURATION AND SERVICE STATUS
    # ----------------------------------------------

    if st.session_state.get("index_error"):
        st.error(
            "The semantic index is unavailable. "
            "Saved tasks are still visible from SQLite, "
            "but AI operations are temporarily disabled."
        )

        if st.button("Retry index initialization"):
            st.session_state.pop("index_error", None)
            st.session_state.pop("service", None)
            st.session_state.pop("graph", None)
            st.session_state.pop("graph_error", None)
            st.rerun()

    if not settings.provider_key_configured:
        st.warning(
            f"AI chat is disabled: no API key found for "
            f"the selected provider "
            f"({settings.llm_provider}). "
            "Add its key to your .env file and restart "
            "Streamlit. Saved tasks remain available."
        )

    elif st.session_state.get("graph_error"):
        st.error(
            "The AI workflow could not start. "
            "Check the provider/model configuration "
            "and server terminal, then restart Streamlit."
        )

    # ----------------------------------------------
    # RESPONSIVE TWO-COLUMN LAYOUT
    # ----------------------------------------------

    chat_col, tasks_col = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    # ----------------------------------------------
    # LEFT COLUMN: CHAT
    # ----------------------------------------------

    with chat_col:

        queued_prompt = st.session_state.pop("queued_prompt", None)

        heading_col, clear_col = st.columns([3, 1])

        with heading_col:
            st.markdown(
                '<div class="tm-h">Conversation</div>',
                unsafe_allow_html=True,
            )

        with clear_col:
            if st.button(
                "Clear chat",
                use_container_width=True,
            ):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": WELCOME,
                        "kind": "info",
                    }
                ]

                st.session_state.thread_id = str(
                    uuid4()
                )

                st.session_state.awaiting_clarification = False

                st.rerun()

        if st.session_state.awaiting_clarification:
            st.warning(
                "Clarification needed — reply to the "
                "last assistant question with an option "
                "number, task ID, or requested date."
            )

        with st.container(
            height=CHAT_HEIGHT - 70,
            border=True,
        ):
            for message in st.session_state.messages:
                render_message(message)

        # Starter prompts: only on a fresh conversation.
        if (
            len(st.session_state.messages) == 1
            and not st.session_state.awaiting_clarification
        ):
            chip_cols = st.columns(len(STARTER_PROMPTS))
            for col, text in zip(chip_cols, STARTER_PROMPTS):
                with col:
                    st.button(
                        text,
                        key=f"starter_{text}",
                        use_container_width=True,
                        disabled=graph is None,
                        on_click=queue_prompt,
                        args=(text,),
                    )

        typed = st.chat_input(
            "Describe a task operation or answer a clarification...",
            disabled=graph is None,
            key="task_chat_input",
        )

        prompt = typed if (typed and typed.strip()) else queued_prompt

        if prompt and prompt.strip() and graph is not None:

            with st.spinner(
                "Understanding and validating your request..."
            ):
                handle_prompt(
                    prompt.strip(),
                    graph,
                )

            # Re-render conversation and reload tasks
            # after the graph has completed.
            st.rerun()

    # ----------------------------------------------
    # RIGHT COLUMN: TASKS
    # ----------------------------------------------

    with tasks_col:
        show_task_list(
            service,
            database,
        )

    st.caption(
        "Local single-user prototype · "
        "Task data persists in SQLite. "
        "Conversation history stays in this browser session."
    )


if __name__ == "__main__":
    main()