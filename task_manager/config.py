"""Safe environment-based settings for the AI Task Manager.

No model calls or database initialization occur in this module.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "openai/gpt-oss-120b"


class Settings(BaseModel):
    """Validated configuration; API keys are obscured by SecretStr."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_provider: Literal["groq", "google"] = "groq"
    llm_model: str = Field(default=DEFAULT_MODEL, min_length=1)
    groq_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    app_timezone: Literal["Asia/Dhaka"] = "Asia/Dhaka"
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def sqlite_path(self) -> Path:
        """Location to use for SQLite in Stage 2 (not created here)."""
        return self.data_dir / "tasks.sqlite3"

    @property
    def chroma_path(self) -> Path:
        """Location to use for ChromaDB in Stage 3 (not created here)."""
        return self.data_dir / "chroma"

    @property
    def provider_key_configured(self) -> bool:
        """Whether the selected provider has a non-empty key."""
        key = self.groq_api_key if self.llm_provider == "groq" else self.google_api_key
        return key is not None and bool(key.get_secret_value())


def get_settings() -> Settings:
    """Load .env without overwriting environment variables; validate values.

    Paths such as DATA_DIR are resolved relative to the project folder,
    regardless of the terminal's current working directory.
    """
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    raw_dir = Path(os.getenv("DATA_DIR", "data")).expanduser()
    data_dir = (raw_dir if raw_dir.is_absolute() else PROJECT_ROOT / raw_dir).resolve()

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip() or None,
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip() or None,
        app_timezone=os.getenv("APP_TIMEZONE", "Asia/Dhaka").strip(),
        data_dir=data_dir,
    )
