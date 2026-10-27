# AI Task Manager — Project Specification

## 1. Goal and scope

Build a single-page, chat-first task manager. A user describes intended task changes in natural language; an LLM proposes a **structured action plan**. Trusted Python code validates and executes it.

**Stage 1 scope:** folders, dependencies, safe configuration, specifications, placeholder interface and basic tests only. No working task database, agent, search index or task mutation yet.

## 2. Technology

- Python 3.12, Streamlit, LangGraph, Pydantic, python-dotenv and pytest.
- SQLite is the **only authoritative source** for task records.
- ChromaDB is a **rebuildable, derived** vector index using its local default embedding function initially.
- A configurable Groq or Google Gemini API is used for language understanding, not direct task execution.
- Git tracks code and configuration examples, never credentials or runtime data.

## 3. Planned task contract

Each task contains:

| Field | Type / restriction |
| --- | --- |
| `id` | SQLite `INTEGER PRIMARY KEY AUTOINCREMENT` (internal) |
| `task_id` | Display form of SQLite `id`, e.g. `TSK-000001` (computed, not LLM-generated) |
| `title` | Required, validated nonempty string |
| `description` | Optional text |
| `status` | `Todo`, `In progress`, `Done` |
| `priority` | `Low`, `Medium`, `High` |
| `due_date` | Optional `YYYY-MM-DD` date (nullable) |
| `created_at`, `updated_at` | Timezone-aware timestamps, persisted in UTC |

Task identifiers remain stable and are not reused after deletion. `task_id` can be formatted on read and need not be stored as a duplicate column.

## 4. Required capabilities (future stages)

1. Create tasks.
2. List tasks and filter by status and/or priority.
3. Semantic task search.
4. Edit titles and descriptions.
5. Move tasks between allowed statuses.
6. Change priority.
7. Set, change or clear due dates.
8. Complete tasks.
9. Delete tasks.
10. Parse and execute multiple actions from one message.
11. Request clarification for ambiguous or missing mutation targets.
12. Show accurate success, failure and clarification results.
13. Persist records after refresh or app restart.

## 5. Logical architecture

```text
Streamlit (one page)
    |
    v
LangGraph orchestration
    |
    v
Provider LLM (Groq / Google) -> Pydantic action-plan validation
    |
    v
Target resolver (ID -> unique exact title -> semantic candidates)
    |
    v
Batch validation (all actions valid and targets unambiguous?)
    |
    v
Allowlisted Task Service -> atomic SQLite transaction [source of truth]
    |
    v
ChromaDB refresh/reconciliation [derived index]
    |
    v
Verified operation results -> Streamlit response
```

For queries, SQLite supplies authoritative task fields; ChromaDB only proposes matching IDs. The service re-reads those IDs from SQLite and filters out missing/deleted entries.

## 6. Data and control flow

**Example user instruction:** “Mark the budget task done and make the presentation high priority.”

1. Streamlit passes chat text to LangGraph.
2. The LLM returns structured proposed actions, **never SQL or Python code**.
3. Pydantic validates the action format, allowed values and field constraints.
4. The resolver looks up identifiers in SQLite; ChromaDB can supply semantic candidates.
5. If zero or several credible targets remain for a mutation, stop and ask a specific clarification question.
6. Resolve and validate **every** action before starting writes. Recheck relevant state inside the write transaction to protect against stale resolutions.
7. Apply the whole batch in **one SQLite transaction**; roll back on any failure. No partial task mutations.
8. After commit, update the Chroma index; retry or rebuild if indexing fails. A vector-index failure must not falsely roll back already committed task state or imply the SQL transaction failed.
9. Present results **only after service confirmation**. Report an indexing warning separately if necessary.

Clarifications must preserve the pending intent without executing it; after the user chooses a target, the batch is revalidated before committing. Read-only list/search requests do not need mutation-target resolution.

## 7. Safety rules and invariants

- SQLite exclusively owns task state; vector search is advisory.
- The LLM never directly executes SQL, code, filesystem writes or database operations.
- Only explicit, allowlisted service functions can mutate tasks.
- Validate lengths, required fields, statuses, priorities, due dates and action schemas before writing.
- Resolve mutations using an explicit task ID first, otherwise a unique exact title, otherwise semantic candidates backed by SQLite. Never guess when an ID is unknown or titles/candidates are ambiguous.
- If any mutation in a multi-action request is invalid or ambiguous, write **nothing**.
- Use `datetime.now(ZoneInfo("Asia/Dhaka")).date()` when resolving relative dates; never rely on the LLM's current date. Convert persisted timestamps to UTC.
- Accept only the defined statuses and priorities; reject unsupported operations.
- Do not claim success until the task service has committed and returned results.
- Handle Chroma synchronization with a repair/rebuild mechanism; do not present its data as authoritative.
- Keep API keys in untracked `.env` or process environment variables, never in tracked code or logs.
- Use parameterized SQL for future database access.

## 8. Local persistence (future stages)

- `data/tasks.sqlite3`: persistent SQLite DB.
- `data/chroma/`: persistent ChromaDB index.
- The entire `data/` directory is ignored by Git except the placeholder `.gitkeep` file.
- Stage 1 does **not** create database tables or vector collections.

## 9. Implementation roadmap

### Stage 1 — Project foundation (this delivery)
- Scaffold package and modules.
- Environment variables, Pydantic settings and safe secret handling.
- Placeholder Streamlit page, requirements, documentation, tests, Git ignores.

### Stage 2 — SQLite and models
- Task/action schemas; database connection and migrations/schema creation.
- SQLite AUTOINCREMENT IDs, CRUD task service, date and enum validation.
- Service-focused tests including restart persistence and rollback.

### Stage 3 — Semantic search
- Chroma local persistence and default embeddings.
- Index updates on create/edit/delete; rebuild from SQLite; candidate lookup.
- Tests for missing/stale index entries.

### Stage 4 — Structured LLM planning
- Provider adapters for Groq/Google and validated action plans.
- Relative date handling in Python, model error and malformed-output handling.
- No execution permitted from unvalidated plans.

### Stage 5 — LangGraph orchestration
- State, parse -> validate -> resolve -> clarify/execute workflow.
- All-or-nothing multi-action mutations; follow-up clarification handling.
- Tests for ambiguous titles, missing IDs, mixed actions and rollback.

### Stage 6 — Finished Streamlit UI and reliability
- Chat, task list/filter, semantic search, confirmations and operation feedback.
- Integration tests, error handling, logging without secrets, version pinning.

## 10. Stage 1 acceptance checks

- Python 3.12 virtual environment can be created and dependencies installed.
- `python -m pytest -q` passes basic configuration and structure tests.
- `python -m streamlit run app.py` starts a placeholder UI without an API key.
- `.env`, `data/` contents, `.venv/` and caches are excluded by `.gitignore`.
- No actual task operations or model calls take place in Stage 1.
