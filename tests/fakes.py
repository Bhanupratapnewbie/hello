"""Test doubles for model and Telegram clients."""

from __future__ import annotations

from command_center.models import ChatMessage


class FakeModelClient:
    """Returns scripted responses in order; records calls for assertions."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def complete(self, model, messages, *, temperature=0.2, max_tokens=None) -> str:
        self.calls.append((model, messages))
        if self.responses:
            return self.responses.pop(0)
        return ""

    async def aclose(self) -> None:
        return None


class FakeTelegramClient:
    """Serves prepared update batches and records sent messages."""

    def __init__(self, update_batches: list[list[dict]] | None = None) -> None:
        self.update_batches = list(update_batches or [])
        self.sent: list[tuple[int, str]] = []

    async def get_updates(self, offset=None, timeout=30) -> list[dict]:
        if self.update_batches:
            return self.update_batches.pop(0)
        return []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((int(chat_id), text))

    async def aclose(self) -> None:
        return None
