"""Configuration for the Command Center.

All values are loaded from environment variables (or a local .env file). The
model layer is intentionally provider-agnostic: every role can be pointed at any
OpenAI-compatible endpoint and any model id, so the operator chooses exactly
which models (including open / uncensored ones) back each role.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Telegram gateway ---
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_admin_chat_id: int = Field(default=0, alias="TELEGRAM_ADMIN_CHAT_ID")

    # --- Model provider (OpenAI-compatible) ---
    model_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="MODEL_BASE_URL"
    )
    # Falls back to OPENROUTER_API_KEY if MODEL_API_KEY is unset.
    model_api_key: str = Field(default="", alias="MODEL_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # --- Role -> model routing ---
    model_planner: str = Field(
        default="nousresearch/hermes-3-llama-3.1-70b", alias="MODEL_PLANNER"
    )
    model_reasoner: str = Field(default="deepseek/deepseek-r1", alias="MODEL_REASONER")
    model_coder: str = Field(
        default="qwen/qwen-2.5-coder-32b-instruct", alias="MODEL_CODER"
    )
    model_fixer: str = Field(
        default="meta-llama/llama-3.1-8b-instruct", alias="MODEL_FIXER"
    )

    # --- Execution environment ---
    workdir: str = Field(default="./workspace", alias="CC_WORKDIR")
    command_timeout: int = Field(default=300, alias="CC_COMMAND_TIMEOUT")
    max_heal_attempts: int = Field(default=3, alias="CC_MAX_HEAL_ATTEMPTS")

    @property
    def effective_api_key(self) -> str:
        """The API key actually used for model calls."""
        return self.model_api_key or self.openrouter_api_key

    def role_model(self, role: str) -> str:
        """Map a logical role name to a configured model id."""
        mapping = {
            "planner": self.model_planner,
            "reasoner": self.model_reasoner,
            "coder": self.model_coder,
            "fixer": self.model_fixer,
        }
        if role not in mapping:
            raise KeyError(f"unknown model role: {role!r}")
        return mapping[role]

    def validate_runtime(self) -> list[str]:
        """Return a list of human-readable problems that block running for real."""
        problems: list[str] = []
        if not self.telegram_bot_token:
            problems.append("TELEGRAM_BOT_TOKEN is not set")
        if not self.telegram_admin_chat_id:
            problems.append("TELEGRAM_ADMIN_CHAT_ID is not set")
        if not self.effective_api_key:
            problems.append("MODEL_API_KEY / OPENROUTER_API_KEY is not set")
        return problems


def load_settings() -> Settings:
    return Settings()
