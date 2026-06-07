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
    # Shared invite token. When set, a non-admin who sends ``/start <token>`` is
    # added as an additional administrator. This lets the owner grant access to a
    # trusted collaborator without giving up their own admin rights (the bot
    # supports multiple admins; all of them receive output and can drive tasks).
    admin_claim_token: str = Field(default="", alias="CC_ADMIN_CLAIM_TOKEN")

    # --- Default model provider (OpenAI-compatible) ---
    # The global endpoint used by any role that does not set its own override
    # below. In the recommended setup this points at the cloud GPU box serving the
    # uncensored open-weight base model (e.g. Ollama/vLLM at http://host:11434/v1).
    model_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="MODEL_BASE_URL"
    )
    # Falls back to OPENROUTER_API_KEY if MODEL_API_KEY is unset.
    model_api_key: str = Field(default="", alias="MODEL_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")

    # --- Role -> model routing ---
    # The base/brain (router). The query first hits this model: the latest-class,
    # fully uncensored open-weight Qwen (Qwen3 abliterated, 32B dense — the largest
    # dense Qwen3; the only larger Qwen3 is the 235B MoE). Run full-precision (no
    # quantization) on a cloud GPU box. Swap for any latest uncensored Qwen/Gemma.
    model_planner: str = Field(
        default="huihui-ai/Qwen3-32B-abliterated", alias="MODEL_PLANNER"
    )
    # Deep research / long-context reasoning. A frontier trained model gives the
    # richest research output here (the uncensored base still routes the query).
    model_reasoner: str = Field(
        default="google/gemini-2.5-pro", alias="MODEL_REASONER"
    )
    # Pure code generation: a top-tier coding model.
    model_coder: str = Field(
        default="anthropic/claude-opus-4.1", alias="MODEL_CODER"
    )
    # Low-latency role: a fast frontier model for rapid error correction.
    model_fixer: str = Field(
        default="google/gemini-2.5-flash", alias="MODEL_FIXER"
    )

    # --- Per-role provider overrides ---
    # Each role can hit its own OpenAI-compatible endpoint + key. When an override
    # is empty it falls back to the global MODEL_BASE_URL / MODEL_API_KEY above.
    # This lets the base/brain run on a local GPU (uncensored open weight) while
    # the research/code/fix roles call cloud APIs (e.g. Gemini, Claude) — typically
    # all fronted by a single OpenRouter key.
    model_planner_base_url: str = Field(default="", alias="MODEL_PLANNER_BASE_URL")
    model_planner_api_key: str = Field(default="", alias="MODEL_PLANNER_API_KEY")
    model_reasoner_base_url: str = Field(default="", alias="MODEL_REASONER_BASE_URL")
    model_reasoner_api_key: str = Field(default="", alias="MODEL_REASONER_API_KEY")
    model_coder_base_url: str = Field(default="", alias="MODEL_CODER_BASE_URL")
    model_coder_api_key: str = Field(default="", alias="MODEL_CODER_API_KEY")
    model_fixer_base_url: str = Field(default="", alias="MODEL_FIXER_BASE_URL")
    model_fixer_api_key: str = Field(default="", alias="MODEL_FIXER_API_KEY")

    # --- Execution environment ---
    workdir: str = Field(default="./workspace", alias="CC_WORKDIR")
    command_timeout: int = Field(default=300, alias="CC_COMMAND_TIMEOUT")
    max_heal_attempts: int = Field(default=3, alias="CC_MAX_HEAL_ATTEMPTS")

    # --- Progress supervisor (watchdog) ---
    # A meta-layer that watches the whole task: it caps the total number of steps,
    # detects when the same action repeats (a loop), and periodically asks the
    # uncensored base model to judge whether the task is still on the right path —
    # continuing, re-planning, asking the admin, or aborting accordingly.
    supervisor_enabled: bool = Field(default=True, alias="CC_SUPERVISOR")
    # Hard ceiling on total executed steps for one request (loop/runaway guard).
    max_total_steps: int = Field(default=40, alias="CC_MAX_TOTAL_STEPS")
    # Run a base-model on-track check every N completed steps (0 disables checks).
    supervisor_interval: int = Field(default=4, alias="CC_SUPERVISOR_INTERVAL")
    # Number of identical consecutive actions that counts as a stuck loop.
    loop_threshold: int = Field(default=3, alias="CC_LOOP_THRESHOLD")
    # Max times the supervisor may re-plan the remaining work before giving up.
    max_replans: int = Field(default=2, alias="CC_MAX_REPLANS")

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

    def role_endpoint(self, role: str) -> tuple[str, str]:
        """Resolve (base_url, api_key) for a role, falling back to the global ones.

        Lets each role target its own provider (e.g. an uncensored open-weight base
        on a local GPU for ``planner`` while ``reasoner``/``coder``/``fixer`` call
        cloud APIs), while keeping single-endpoint setups working unchanged.
        """
        overrides = {
            "planner": (self.model_planner_base_url, self.model_planner_api_key),
            "reasoner": (self.model_reasoner_base_url, self.model_reasoner_api_key),
            "coder": (self.model_coder_base_url, self.model_coder_api_key),
            "fixer": (self.model_fixer_base_url, self.model_fixer_api_key),
        }
        if role not in overrides:
            raise KeyError(f"unknown model role: {role!r}")
        base_url, api_key = overrides[role]
        return (base_url or self.model_base_url, api_key or self.effective_api_key)

    def validate_runtime(self) -> list[str]:
        """Return a list of human-readable problems that block running for real."""
        # TELEGRAM_ADMIN_CHAT_ID is intentionally optional: when unset (0) the
        # bot auto-claims the first /start sender as the administrator.
        problems: list[str] = []
        if not self.telegram_bot_token:
            problems.append("TELEGRAM_BOT_TOKEN is not set")
        if not self.effective_api_key:
            problems.append("MODEL_API_KEY / OPENROUTER_API_KEY is not set")
        return problems


def load_settings() -> Settings:
    return Settings()
