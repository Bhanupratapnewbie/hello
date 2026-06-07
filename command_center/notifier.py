"""Notifier abstraction.

The brain and executor stream their progress through an async ``Notifier`` so the
core logic stays decoupled from Telegram. Production wires this to the Telegram
gateway; tests use ``CollectingNotifier`` to assert on emitted messages.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# An async sink for streamed log/progress lines.
Notifier = Callable[[str], Awaitable[None]]


class CollectingNotifier:
    """Notifier that stores messages in memory (used in tests and dry runs)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def __call__(self, text: str) -> None:
        self.messages.append(text)

    @property
    def transcript(self) -> str:
        return "\n".join(self.messages)


async def noop_notifier(_text: str) -> None:
    """A notifier that discards everything."""
    return None
