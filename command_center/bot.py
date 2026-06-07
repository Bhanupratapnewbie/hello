"""The Telegram gateway: the sole control interface for the Command Center.

Responsibilities:
* long-poll Telegram for the administrator's messages
* enforce single-administrator access (auto-claimed on first /start)
* stream every brain/executor log line back to the administrator
* surface clarification questions and feed the reply back into the brain
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .brain import Brain
from .config import Settings
from .executor import ShellExecutor
from .models import ModelClient
from .telegram import TelegramClient, TelegramError

_HELP = (
    "Command Center online.\n"
    "Send any task in plain language and I will plan, route, execute, and self-heal.\n"
    "Prefix a message with ! to run it as a raw shell command immediately.\n"
    "Commands: /start bind admin, /status show state, /cancel abort current task."
)


class CommandCenterBot:
    def __init__(
        self,
        settings: Settings,
        *,
        telegram: TelegramClient | None = None,
        model_client: ModelClient | None = None,
    ) -> None:
        self.settings = settings
        self.telegram = telegram or TelegramClient(settings.telegram_bot_token)
        self.model_client = model_client or ModelClient(
            settings.model_base_url, settings.effective_api_key
        )
        self._state_path = Path(settings.workdir).expanduser().resolve() / ".cc_admin"
        self.admin_id: int = settings.telegram_admin_chat_id or self._load_admin_id()

        self._pending_clarification: asyncio.Future[str] | None = None
        self._task: asyncio.Task | None = None
        self._offset: int | None = None
        self._running = False

        self.executor = ShellExecutor(
            workdir=settings.workdir,
            timeout=settings.command_timeout,
            max_heal_attempts=settings.max_heal_attempts,
            model_client=self.model_client,
            fixer_model=settings.role_model("fixer"),
            notify=self._notify,
        )
        self.brain = Brain(
            settings=settings,
            model_client=self.model_client,
            executor=self.executor,
            notify=self._notify,
            ask_admin=self._ask_admin,
        )

    # --- admin persistence ---
    def _load_admin_id(self) -> int:
        try:
            return int(self._state_path.read_text().strip())
        except (OSError, ValueError):
            return 0

    def _save_admin_id(self, chat_id: int) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(str(chat_id))
        except OSError:
            pass

    # --- messaging ---
    async def _notify(self, text: str) -> None:
        if not self.admin_id:
            return
        try:
            await self.telegram.send_message(self.admin_id, text)
        except TelegramError:
            pass

    async def _ask_admin(self, question: str) -> str:
        loop = asyncio.get_running_loop()
        self._pending_clarification = loop.create_future()
        await self._notify(f"clarification needed: {question}")
        try:
            return await self._pending_clarification
        finally:
            self._pending_clarification = None

    # --- dispatch ---
    async def _handle_message(self, chat_id: int, text: str) -> None:
        text = text.strip()

        # Resolve a pending clarification first (only from the admin).
        if (
            chat_id == self.admin_id
            and self._pending_clarification is not None
            and not self._pending_clarification.done()
        ):
            self._pending_clarification.set_result(text)
            return

        # Admin auto-claim: first /start binds the administrator.
        if not self.admin_id:
            if text.startswith("/start"):
                self.admin_id = chat_id
                self._save_admin_id(chat_id)
                await self.telegram.send_message(chat_id, _HELP)
            else:
                await self.telegram.send_message(
                    chat_id, "Send /start to claim this Command Center."
                )
            return

        # Enforce single-admin access.
        if chat_id != self.admin_id:
            return

        if text.startswith("/start") or text.startswith("/help"):
            await self._notify(_HELP)
            return
        if text.startswith("/status"):
            busy = self._task is not None and not self._task.done()
            await self._notify(f"admin bound. busy={busy}.")
            return
        if text.startswith("/cancel"):
            if self._task is not None and not self._task.done():
                self._task.cancel()
                await self._notify("current task cancelled")
            else:
                await self._notify("nothing to cancel")
            return

        if self._task is not None and not self._task.done():
            await self._notify("busy with a task; send /cancel to abort it first")
            return

        self._task = asyncio.create_task(self._run_task(text))

    async def _run_task(self, text: str) -> None:
        try:
            await self.brain.handle(text)
        except asyncio.CancelledError:
            await self._notify("task aborted")
            raise
        except Exception as exc:  # surface, never crash the gateway
            await self._notify(f"unexpected error: {exc!r}")

    @staticmethod
    def _extract(update: dict) -> tuple[int, str] | None:
        message = update.get("message") or update.get("edited_message")
        if not message:
            return None
        text = message.get("text")
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        if text is None or chat_id is None:
            return None
        return int(chat_id), str(text)

    async def poll_once(self) -> None:
        updates = await self.telegram.get_updates(offset=self._offset, timeout=30)
        for update in updates:
            self._offset = int(update["update_id"]) + 1
            parsed = self._extract(update)
            if parsed is None:
                continue
            await self._handle_message(*parsed)

    async def run(self) -> None:
        self._running = True
        await self._notify("Command Center started.")
        while self._running:
            try:
                await self.poll_once()
            except TelegramError:
                await asyncio.sleep(3)
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        self._running = False
        await self.telegram.aclose()
        await self.model_client.aclose()
