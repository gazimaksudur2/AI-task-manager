"""Verify that actual Git behavior excludes local state and credentials."""
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_git_ignore_rules(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git not installed")
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    paths = [
        ".env", ".env.production", "data/tasks.sqlite3",
        "data/chroma/chroma.sqlite3", ".venv/pyvenv.cfg",
        "venv/bin/python", "FULL_PROJECT_FOR_AI.txt",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == set(paths)
    # Intentionally keep the example settings and empty data marker trackable.
    included = subprocess.run(
        ["git", "check-ignore", "--no-index", ".env.example", "data/.gitkeep"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert included.returncode == 1
