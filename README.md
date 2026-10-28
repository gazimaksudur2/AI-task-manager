# AI Task Manager

A local, single-user task manager for creating, finding, updating, completing, and deleting tasks with natural-language chat. The Streamlit page includes the conversation, task board, status/priority filters, and refresh control.

## Features

- Task IDs (`TSK-000001`), descriptions, priorities, statuses, and due dates.
- Natural-language commands, semantic search, and multiple actions in one request.
- Clarification for missing or ambiguous targets; invalid plans are rejected before changes are made.
- SQLite persistence and a rebuildable ChromaDB search index.
- OpenAI, Google Gemini, Groq, or Anthropic as the LLM provider.

The LLM only proposes structured actions. Pydantic and Python validate them; SQLite is the source of truth, while ChromaDB is a derived search index. Multi-action mutations are validated as a batch and committed in one SQLite transaction.

## Requirements

- **Python 3.12** (recommended project version), `pip`, and `venv`.
- An API key for one supported LLM provider to use AI chat.
- Internet access for installing packages, provider requests, and the first ChromaDB embedding-model download if needed.

No separate database server is required; SQLite is included with Python.

## Setup on Ubuntu / Linux

Open a terminal in the project directory—the folder containing `app.py` and `requirements.txt`. If `python3` is unavailable, install Python 3.12 and its `venv` package for your Ubuntu release first.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

If `.env` already exists, edit it rather than overwriting it. In VS Code, open the project folder and select `.venv/bin/python` with **Python: Select Interpreter**.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Configure the provider

Edit `.env` and set **one matching provider/model pair** and its API key. The template selects OpenAI but leaves the key blank. Model names below are examples; use a model currently available to your account.

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=
GROQ_API_KEY=
CLAUDE_API_KEY=
APP_TIMEZONE=Asia/Dhaka
DATA_DIR=data
MAX_SEMANTIC_DISTANCE=
```

| Provider | `LLM_PROVIDER` | Example model | Key variable |
| --- | --- | --- | --- |
| OpenAI | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Google Gemini | `google` | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| Groq | `groq` | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| Anthropic | `anthropic` | `claude-sonnet-4-6` | `CLAUDE_API_KEY` |

Keep unused keys blank. `APP_TIMEZONE` is currently restricted to `Asia/Dhaka`; leave `MAX_SEMANTIC_DISTANCE` blank unless you have calibrated a cutoff for your embedding model. Never commit or share `.env`. Provider API usage may incur separate charges.

Shell environment variables override `.env`; if the wrong provider/model is selected, check whether `LLM_PROVIDER` or `LLM_MODEL` is already exported. Restart Streamlit after configuration changes.

## Run the app

With the virtual environment active:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`. Stop the server with **Ctrl+C**. Without a provider key, the page can display saved tasks from SQLite, but AI chat is disabled.

## Example commands

- `Create a high-priority task to prepare my demo tomorrow.`
- `Show all my tasks.`
- `Show unfinished tasks related to documentation.`
- `Move TSK-000001 to In progress.`
- `Change TSK-000001 priority to Low.`
- `Mark TSK-000001 done.`
- `Delete TSK-000001.`

Statuses: `Todo`, `In progress`, `Done`. Priorities: `Low`, `Medium`, `High`. Due dates support `YYYY-MM-DD`, `today`, `tomorrow`, `day after tomorrow`, `in N days`, and supported weekday phrases (`next Friday` uses the following-week convention). For ambiguous targets, identify the task by ID or unique title rather than asking the agent to guess.

## Tests

Run from the project root:

```bash
python -m pytest -q
python -m pytest tests/test_end_to_end.py -v
python -m compileall -q app.py task_manager tests
```

Automated agent tests use fake planners and do not require live LLM calls. A real-provider smoke test needs a valid key, an available model, network access, and account quota.

## Data and recovery

- `data/tasks.sqlite3` — authoritative task records, revisions, index queue, and mutation receipts.
- `data/chroma/` — rebuildable semantic search index.
- `.env` — provider credentials and settings; `.venv/` — virtual environment.

These runtime files are excluded by `.gitignore`. `DATA_DIR` can change the data location; relative paths resolve from the project root. Back up SQLite with the app stopped. **Do not delete SQLite to fix ChromaDB.** If search results are stale, stop Streamlit and rebuild the index from SQLite:

```bash
python - <<'PY'
from task_manager.config import get_settings
from task_manager.database import TaskDatabase
from task_manager.vector_store import TaskVectorStore
from task_manager.task_service import TaskService

cfg = get_settings()
db = TaskDatabase(cfg.sqlite_path)
index = TaskVectorStore(cfg.chroma_path, max_distance=cfg.max_semantic_distance)
service = TaskService(db, index)
print("Reconciled (upserts, deletions):", service.reconcile_index())
print("Pending index changes:", db.pending_index_changes())
PY
```

## Troubleshooting

- **`ModuleNotFoundError`:** activate `.venv` and reinstall with `python -m pip install -r requirements.txt`.
- **AI chat disabled:** add the selected provider's key, check the provider/model pair, and restart Streamlit.
- **Provider/model error:** verify model access, connectivity, quota, and billing.
- **ChromaDB fails to initialize:** check directory permissions and internet access for the first embedding download; then use **Retry index initialization** in the UI.
- **Task saved but search misses it:** check the task board, then run the reconciliation command above; SQLite may have committed before index synchronization failed.
- **Chat history disappears after reload:** expected; tasks persist, but conversation and clarification state are held in memory.

## Project structure

```text
app.py
.streamlit/config.toml         # UI theme
.env.example                   # Safe configuration template
requirements.txt               # Python dependencies
PROJECT_SPEC.md                # Design and implementation history
project_reader.txt             # Exports source files to one text file
demo_agent.py                  # Optional CLI using a real LLM provider
demo_data_layer.py             # Optional demo that writes sample tasks
task_manager/
  config.py                    # Environment settings
  models.py                    # Task models and ID validation
  agent_models.py              # Structured action schemas
  agent.py                     # LLM adapters and planner
  graph.py                     # LangGraph workflow
  intent_resolution.py         # Request/target resolution
  database.py                  # SQLite persistence
  task_service.py              # Task operations and index sync
  vector_store.py              # ChromaDB semantic search
tests/                         # Automated unit and integration tests
```

## Limitations

This is a local, single-user prototype with no authentication, role separation, encryption, cloud backups, or production deployment safeguards. Chat history and clarification checkpoints are in memory, and concurrent app processes are not coordinated for ChromaDB writes. Run one Streamlit process and do not expose the app publicly without adding appropriate security and operational controls.
