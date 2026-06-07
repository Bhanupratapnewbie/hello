"""Provider-agnostic model client.

Speaks the OpenAI-compatible Chat Completions protocol, so it works against
OpenRouter, OpenAI, Together, vLLM, Ollama (``/v1``), and similar endpoints.
The brain selects which configured model id to use per role; this client only
cares about "call this model id at this base url".
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class ModelError(RuntimeError):
    """Raised when a model endpoint returns an error or malformed response."""


@dataclass
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ModelClient:
    """Thin async wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """Return the assistant message content for a chat completion.

        ``base_url`` / ``api_key`` override the client defaults for this call,
        which is how the brain points each role at its own provider.
        """
        client = await self._ensure_client()
        target_base = (base_url or self._base_url).rstrip("/")
        target_key = api_key or self._api_key
        payload: dict[str, object] = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {target_key}",
            "Content-Type": "application/json",
            # Optional attribution headers used by some gateways (e.g. OpenRouter).
            "HTTP-Referer": "https://github.com/Bhanupratapnewbie/hello",
            "X-Title": "Command Center",
        }

        try:
            resp = await client.post(
                f"{target_base}/chat/completions",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:  # network-level failure
            raise ModelError(f"request to model endpoint failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ModelError(
                f"model endpoint returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ModelError(f"malformed model response: {exc}: {resp.text[:500]}") from exc
