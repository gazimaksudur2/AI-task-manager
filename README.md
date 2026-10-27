# AI Task Manager

Single-page, **local single-user** task manager with natural-language commands. The application uses Streamlit for the interface, LangGraph for orchestration, a configurable LLM to **propose** structured actions, Pydantic for validation, SQLite for durable task records, and ChromaDB for semantic task search. The model never executes SQL or filesystem writes.

**Status:** Stage 4 implemented; backend integration and real-provider smoke tests must be run in the target Ubuntu/VS Code environment before final deployment. The project is **not** designed for public or multi-user hosting.

## Architecture

```text
Streamlit chat + task board
        |
        v
LangGraph: understand -> validate/resolve -> clarify OR execute -> respond
        |                       |
        v                       v
Groq / Gemini / OpenAI /        TaskService (allowlisted mutations)
Anthropic structured planner      |                 |
                                 SQLite             ChromaDB
                              source of truth     derived vector index
                                 |                 ^
                                 +-- sync queue ---+
```

- **SQLite** contains authoritative tasks, numeric AUTOINCREMENT task keys displayed as `TSK-000001`, revision numbers, a durable Chroma synchronization queue, and idempotency receipts. Deleting a task never reuses its identifier.
- **ChromaDB** embeds the task's title and description so different phrases can retrieve related tasks. It is **not** authoritative; candidate IDs are re-read from SQLite. Its local default embedding model may download model weights at first use and require network access. A deleted task cannot be recreated from the vector index.
- **Embeddings** are numeric representations of text: similar task wording tends to have nearby vectors. Semantic search ranks approximate matches and may return irrelevant entries. `MAX_SEMANTIC_DISTANCE` optionally filters results; it is intentionally unset until calibrated for the selected model and distance metric.
- **LangGraph** maintains the request/clarification workflow for the current app session. Read-only LIST/SEARCH requests do not mutate SQLite; a batch of mutations is prevalidated and committed in one SQLite transaction. Clarifications defer *all* writes until resolved.
- **Reliability:** The app assigns a unique submission ID per chat submission. SQLite atomically saves a receipt for successful mutation batches, so the same submission ID cannot apply a batch twice. Chroma writes occur only after the SQLite commit. If indexing fails, the committed task remains and an index-repair warning is returned.

## Prerequisites

- **Python 3.12**, ideally with `python3.12-venv` and `python3.12-pip` available on Ubuntu.
- Git and VS Code (recommended) with the Microsoft Python extension.
- A usable LLM API key (OpenAI, Groq, Google Gemini, or Anthropic) for AI chat; SQLite read-only display works without it.
- Internet initially for installing packages, optional default Chroma embedding weights and the chosen LLM provider.

## Ubuntu / VS Code setup

Open **File → Open Folder** in VS Code and select the `ai-task-manager` project directory. Open **Terminal → New Terminal**, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the VS Code interpreter using **Python: Select Interpreter → `.venv/bin/python`**. On subsequent sessions run `source .venv/bin/activate`. If your Ubuntu command is `python3`, check `python3 --version` and ensure the version meets the project requirement.

On Windows PowerShell use `py -3.12 -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, then the same pip, copy, test and run commands (`Copy-Item .env.example .env`).

## OpenAI API-key configuration

1. Go to [OpenAI API keys](https://platform.openai.com/api-keys), select the intended API project, and create a secret key. Store it securely; it may be shown only once.
2. Open the local `.env` file (which is ignored by Git) and configure exactly one provider/model pair:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=YOUR_OWN_SECRET_KEY_HERE
APP_TIMEZONE=Asia/Dhaka
DATA_DIR=data
MAX_SEMANTIC_DISTANCE=
```

3. Save `.env`, then restart Streamlit. Use a model ID enabled in *your* API project; model availability, costs, rate limits and method support can change.
4. **API usage and billing are separate from a ChatGPT subscription.** Never paste your real key into a repository, issue, screenshot, or shared project export.

Alternative examples (change **both** provider and model):

```dotenv
# Google Gemini (use GOOGLE_API_KEY=...)
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash

# Groq (use GROQ_API_KEY=...)
# LLM_PROVIDER=groq
# LLM_MODEL=openai/gpt-oss-120b

# Anthropic (use CLAUDE_API_KEY=...)
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-sonnet-4-6
```

Only leave one active `LLM_PROVIDER` and `LLM_MODEL` pair in your actual `.env`. The remaining keys can stay blank. An existing exported shell variable overrides `.env`; for example, check `echo "$LLM_PROVIDER"` if your selected provider seems unexpected. The model is used only for structured intent parsing; it does not directly write tasks.

## Run the application

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`. Keep the development server private. Stop it with Ctrl+C. The page shows chat and task cards side by side on desktops, plus status/priority filters and a refresh control. Descriptions are shown in expandable sections.

### Example natural-language commands

- `Create a high-priority task to prepare my demo tomorrow.`
- `Show all my tasks.`
- `Find unfinished tasks related to documentation.`
- `Move TSK-000001 to In progress.`
- `Change TSK-000001 priority to Low.`
- `Rename TSK-000001 to Final demonstration.`
- `Clear the due date on TSK-000001.`
- `Mark TSK-000001 done.`
- `Delete TSK-000001.`
- `Complete TSK-000001 and set TSK-000002 priority to High.`

Task IDs, explicit unique titles, and confirmed semantic candidates are supported. If a title is shared by multiple tasks, the agent asks which one you mean **without performing any mutations**. Supported statuses: `Todo`, `In progress`, `Done`; supported priorities: `Low`, `Medium`, `High`. Due dates include `YYYY-MM-DD`, `today`, `tomorrow`, `day after tomorrow`, `in N days`, and `this/next weekday`. Other date phrasing requires an exact date confirmation. Relative dates use Asia/Dhaka. `next Friday` follows the implementation's *following-week* convention.

## Tests

```bash
python -m pytest -q
python -m pytest tests/test_end_to_end.py -v
python -m pytest tests/test_database_offline.py tests/test_config.py -q
python -m compileall -q app.py task_manager tests
```

- All model-dependent agent tests use **fake planners**; the test suite does not need or call real LLM APIs. End-to-end integration tests use temporary SQLite and Chroma files, and a deterministic, local test embedder.
- `tests/test_end_to_end.py` covers the requested fourteen scenarios, including persistence, semantic search, multi-field updates, ambiguous-target clarification, atomic batches and simulated failures.
- `tests/test_database_offline.py` and `tests/test_config.py` can run with just `pytest`, `pydantic`, and `python-dotenv` installed; they also test idempotent submissions and rollback after a write is attempted.
- Real UI and provider smoke tests are **manual**. Use the [verification checklist](docs/VERIFICATION.md) for observable steps and expected results. Do not describe a real provider or Streamlit run as verified until performed in the target machine.

## Persistence and recovery

| Data | Default location | Git-tracked? |
| --- | --- | --- |
| SQLite tasks, revisions, index queue, mutation receipts | `data/tasks.sqlite3` | No |
| Chroma collection, embeddings, metadata | `data/chroma/` | No |
| Credentials | `.env` | No |
| Safe configuration template | `.env.example` | Yes |
| Python virtual environment | `.venv/` | No |

The `DATA_DIR` environment variable can change the `data/` location; relative paths resolve from the project root. Chat history and clarification checkpoints are held in process/browser-session memory and may disappear on a full reload or server restart. Saved tasks do not disappear.

**To recover semantic search:** stop Streamlit and run from the project root:

```bash
source .venv/bin/activate
python - <<'PY'
from task_manager.config import get_settings
from task_manager.database import TaskDatabase
from task_manager.vector_store import TaskVectorStore
from task_manager.task_service import TaskService
cfg = get_settings()
service = TaskService(
    TaskDatabase(cfg.sqlite_path),
    TaskVectorStore(cfg.chroma_path, max_distance=cfg.max_semantic_distance),
)
print('Reconciled (upserts, deletions):', service.reconcile_index())
print('Pending index changes:', service.database.pending_index_changes())
PY
```

This rebuilds missing/stale vector records using the authoritative SQLite task table. **Never delete the SQLite database to fix Chroma.** Back up `data/tasks.sqlite3` while Streamlit is stopped before manual maintenance; preserve the SQLite journal/WAL files too if you back it up while it is in use.

## Troubleshooting

| Symptom | Check / action |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv`, ensure VS Code interpreter matches, reinstall `requirements.txt`. |
| AI chat disabled | Set the selected provider's key and restart; SQLite task list is still readable. |
| Provider/model error | Verify matching `LLM_PROVIDER`, `LLM_MODEL`, correct project API key, available model, and account limits; click **Retry AI workflow** after correcting setup. |
| Chroma startup fails | Check embedding weights/network, local permissions and `data/chroma` path; click **Retry index initialization**. |
| Task saved but search missing it | SQLite committed; reconcile the index with the command above. |
| Unexpected task choice | Use a task ID or complete exact title; ambiguous mutations must ask for clarification. |
| Invalid date | Supply `YYYY-MM-DD` when prompted. |
| Old conversation is gone | Expected after full reload/restart: task data remains in SQLite, chat context is session-scoped. |
| Retry after uncertain failure | Check SQLite and the visible task list first; a write may have committed before an index/network error. |
| Same chat submission retried | Streamlit uses a submission ID and SQLite receipts to avoid duplicate batch writes. A **new** deliberately repeated message is a new request. |

## Scope and limitations

This is a local **single-user** application. It has no login, authorization, tenant separation, encrypted storage, cloud backups or public deployment hardening. The semantic index may need first-run model download; relevance depends on embeddings. LangGraph checkpoints are in memory, so clarification conversations are not durable across restarts. The natural-language parser supports a finite set of task operations and date forms; pronoun references such as `finish that` are not yet a general conversation memory feature. Concurrent processes can race on Chroma index writes, so use a single Streamlit server process for this prototype. Idempotency receipts accumulate with task history and may need a controlled retention policy later. Provider behavior depends on model capabilities, API access, quotas and connectivity.

**Possible improvements:** durable LangGraph checkpoints, opt-in task previews for risky batch changes, configurable top-k/relevance diagnostics, background index-sync worker with concurrency locking, task search UI, accessibility/e2e browser testing, backups, authentication and multi-user partitioning.

## Project layout

```text
app.py                        # Streamlit chat/task board
demo_agent.py                 # Interactive CLI for real provider testing
demo_data_layer.py            # Direct data-layer demo
task_manager/config.py        # Environment settings
task_manager/models.py        # Validated task types
task_manager/agent_models.py  # Allowlisted action schemas
task_manager/agent.py         # Provider adapters and LLM planner
task_manager/graph.py         # LangGraph flow and clarification
task_manager/database.py      # SQLite CRUD, transactions and receipts
task_manager/task_service.py  # Authoritative operations and index sync
task_manager/vector_store.py  # ChromaDB search and repair
tests/                       # Automated reliability tests
docs/VERIFICATION.md          # Manual provider/UI smoke checklist
```

See `PROJECT_SPEC.md` for the design invariants, implementation progress and development history.
