"""CLI entrypoint: run the Telegram-controlled Command Center directly.

Usage:
    python -m command_center
"""

from __future__ import annotations

import asyncio
import sys

from .bot import CommandCenterBot
from .config import load_settings


async def _main() -> int:
    settings = load_settings()
    problems = settings.validate_runtime()
    if problems:
        print("Cannot start. Fix configuration:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("See .env.example for all settings.", file=sys.stderr)
        return 1

    bot = CommandCenterBot(settings)
    print("Command Center starting (Telegram gateway). Ctrl+C to stop.")
    try:
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()
    return 0


def run() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    run()
