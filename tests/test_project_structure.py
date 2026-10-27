"""Sanity-check the files and ignore rules in the starter project."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    files = (
        "app.py", "PROJECT_SPEC.md", "README.md", "requirements.txt",
        ".env.example", ".gitignore", "data/.gitkeep",
        "task_manager/__init__.py", "task_manager/config.py",
        "task_manager/models.py", "task_manager/database.py",
        "task_manager/task_service.py", "task_manager/vector_store.py",
        "task_manager/agent.py", "task_manager/graph.py",
    )
    for file in files:
        assert (ROOT / file).is_file(), f"Missing project file: {file}"


def test_gitignore_covers_runtime_secrets_and_files():
    ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in (".env", "!/data/.gitkeep", "/data/**", ".venv/", "__pycache__/", ".pytest_cache/"):
        assert rule in ignore_rules


def test_environment_template_contains_no_keys():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GROQ_API_KEY=\n" in text
    assert "GOOGLE_API_KEY=\n" in text
