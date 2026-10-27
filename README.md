# AI Task Manager — Stage 1

A beginner-friendly, production-minded task manager to be built in stages using Streamlit, LangGraph, ChromaDB, SQLite and a configurable Groq/Google language model.

**Current status:** The UI is a placeholder. Task persistence, semantic search and AI actions are not implemented yet. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full plan.

## Requirements

- Python **3.12** installed and accessible on the terminal
- VS Code with the Python extension
- Git

## Setup — macOS / Linux

```bash
cd ai-task-manager-stage1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m pytest -q
python -m streamlit run app.py
```

## Setup — Windows PowerShell

```powershell
cd ai-task-manager-stage1
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
python -m streamlit run app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for **that terminal session** and then activate; alternatively run `.venv\Scripts\python.exe` directly.

Open the local URL shown by Streamlit (typically `http://localhost:8501`). Stop with **Ctrl+C**.

In VS Code: **File → Open Folder** (the project folder), then **Python: Select Interpreter** and choose the `.venv` Python executable.

## Configuration

Copy `.env.example` to `.env`. The default `LLM_PROVIDER=groq` and `LLM_MODEL=openai/gpt-oss-120b` are only configuration defaults; **Stage 1 does not contact an LLM or require an API key**.

To plan for Google instead, change the environment values to `LLM_PROVIDER=google` and an available Gemini model such as `LLM_MODEL=gemini-2.5-flash`. Set the relevant provider key only when an LLM stage is implemented. Provider model availability and free-tier rules may change.

Environment options:

- `LLM_PROVIDER`: `groq` or `google`
- `LLM_MODEL`: provider model ID, nonempty
- `GROQ_API_KEY`, `GOOGLE_API_KEY`: optional secrets; never commit these
- `APP_TIMEZONE`: must be `Asia/Dhaka`
- `DATA_DIR`: runtime folder, default `data` relative to the project root

The program does not print API keys in the interface. Real secrets go in `.env`, not `.env.example`.

## Tests and Git

```bash
python -m pytest -q
git init
git add .
git status --short
git check-ignore .env data/tasks.sqlite3 data/chroma/index.bin .venv/pyvenv.cfg
```

The last command lists ignored local paths; actual database/vector files should not be created in Stage 1.

## Stage 1 limitations

`task_manager/models.py`, `database.py`, `task_service.py`, `vector_store.py`, `agent.py` and `graph.py` are intentionally documentation-only placeholders. No DB or index is created, and the chat input is disabled. This avoids mixing several learning stages.
