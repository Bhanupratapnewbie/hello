"""Minimal async Telegram Bot API client (long polling).

Deliberately dependency-light (just httpx) so the gateway has no hidden event
loop of its own and integrates cleanly with the rest of the asyncio app.
"""

from __future__ import annotations

import httpx

TELEGRAM_MAX_MESSAGE = 4096


class TelegramError(RuntimeError):
    pass


class TelegramClient:
    def __init__(
        self,
        token: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._client = client

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_updates(
        self, offset: int | None = None, timeout: int = 30
    ) -> list[dict]:
        client = await self._ensure_client()
        params: dict[str, object] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = await client.get(
                f"{self._base}/getUpdates",
                params=params,
                timeout=timeout + 15,
            )
        except httpx.HTTPError as exc:
            raise TelegramError(f"getUpdates failed: {exc}") from exc
        data = resp.json()
        if not data.get("ok"):
            raise TelegramError(f"getUpdates error: {data}")
        return data.get("result", [])

    async def send_message(self, chat_id: int, text: str) -> None:
        client = await self._ensure_client()
        for chunk in _chunk(text, TELEGRAM_MAX_MESSAGE):
            try:
                resp = await client.post(
                    f"{self._base}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk},
                )
            except httpx.HTTPError as exc:
                raise TelegramError(f"sendMessage failed: {exc}") from exc
            data = resp.json()
            if not data.get("ok"):
                raise TelegramError(f"sendMessage error: {data}")


def _chunk(text: str, size: int) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]
