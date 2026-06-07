"""FastAPI wrapper for deployment.

The HTTP surface is intentionally minimal: only health/status endpoints, never a
control interface. Telegram remains the sole gateway for issuing commands. The
Telegram bot runs as a background asyncio task started on app startup, which makes
the whole thing deployable as a standard ASGI service.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .bot import CommandCenterBot
from .config import load_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    problems = settings.validate_runtime()
    bot: CommandCenterBot | None = None
    task: asyncio.Task | None = None

    if problems:
        app.state.status = "misconfigured: " + "; ".join(problems)
    else:
        bot = CommandCenterBot(settings)
        task = asyncio.create_task(bot.run())
        app.state.status = "running"

    app.state.bot = bot
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
        if bot is not None:
            await bot.stop()


app = FastAPI(title="Command Center", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "command-center", "gateway": "telegram"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": getattr(app.state, "status", "unknown")}
