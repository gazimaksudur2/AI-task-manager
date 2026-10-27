"""Check configuration without network access or external services."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import task_manager.config as config
from task_manager.config import PROJECT_ROOT, get_settings


@pytest.fixture(autouse=True)
def clear_settings_env(monkeypatch):
    """Isolate tests from the developer's existing shell configuration."""
    # Unit tests must not depend on the developer's private .env file.
    monkeypatch.setattr(config, "load_dotenv", lambda **kwargs: False)
    for name in (
        "LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "GOOGLE_API_KEY",
        "APP_TIMEZONE", "DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_settings():
    settings = get_settings()
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "openai/gpt-oss-120b"
    assert settings.app_timezone == "Asia/Dhaka"
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.sqlite_path == PROJECT_ROOT / "data" / "tasks.sqlite3"
    assert settings.chroma_path == PROJECT_ROOT / "data" / "chroma"


def test_provider_and_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GOOGLE")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    settings = get_settings()
    assert settings.llm_provider == "google"
    assert settings.llm_model == "gemini-2.5-flash"


def test_secret_is_not_shown_in_repr(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "private-example-key")
    settings = get_settings()
    assert settings.provider_key_configured
    assert "private-example-key" not in repr(settings)
    assert settings.groq_api_key.get_secret_value() == "private-example-key"


def test_missing_key_is_allowed_in_stage_one():
    assert not get_settings().provider_key_configured


def test_reject_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ValidationError):
        get_settings()


def test_reject_blank_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "   ")
    with pytest.raises(ValidationError):
        get_settings()


def test_enforce_dhaka_timezone(monkeypatch):
    monkeypatch.setenv("APP_TIMEZONE", "UTC")
    with pytest.raises(ValidationError):
        get_settings()


def test_relative_data_path_uses_project_root(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "local-data")
    assert get_settings().data_dir == (PROJECT_ROOT / "local-data").resolve()


def test_absolute_data_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert get_settings().data_dir == tmp_path.resolve()


def test_dotenv_values_are_read_without_overriding_shell(monkeypatch, tmp_path: Path):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=google\nLLM_MODEL=gemini-2.5-flash\n",
        encoding="utf-8",
    )
    # Restore the real loader for this integration-style test only.
    from dotenv import load_dotenv as actual_load_dotenv

    monkeypatch.setattr(config, "load_dotenv", actual_load_dotenv)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    settings = get_settings()
    assert settings.llm_provider == "groq"  # shell wins
    assert settings.llm_model == "gemini-2.5-flash"  # .env fallback
