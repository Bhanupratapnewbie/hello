import asyncio

import pytest

from command_center.bot import CommandCenterBot
from command_center.config import Settings
from tests.fakes import FakeModelClient, FakeTelegramClient


def _settings(tmp_path, admin_id=0):
    return Settings(
        telegram_bot_token="t",
        telegram_admin_chat_id=admin_id,
        model_api_key="k",
        workdir=str(tmp_path),
    )


def _bot(tmp_path, admin_id=0, telegram=None):
    return CommandCenterBot(
        _settings(tmp_path, admin_id),
        telegram=telegram or FakeTelegramClient(),
        model_client=FakeModelClient(),
    )


@pytest.mark.asyncio
async def test_auto_claim_admin_on_start(tmp_path):
    tg = FakeTelegramClient()
    bot = _bot(tmp_path, admin_id=0, telegram=tg)
    await bot._handle_message(111, "/start")
    assert bot.admin_id == 111
    # Persisted for restarts.
    assert (tmp_path / ".cc_admin").read_text().strip() == "111"
    assert any(chat == 111 for chat, _ in tg.sent)


@pytest.mark.asyncio
async def test_non_admin_ignored_after_claim(tmp_path):
    tg = FakeTelegramClient()
    bot = _bot(tmp_path, admin_id=111, telegram=tg)
    await bot._handle_message(222, "do something bad")
    assert tg.sent == []
    assert bot._task is None


@pytest.mark.asyncio
async def test_clarification_resolves(tmp_path):
    bot = _bot(tmp_path, admin_id=111)
    loop = asyncio.get_running_loop()
    bot._pending_clarification = loop.create_future()
    await bot._handle_message(111, "the answer")
    assert bot._pending_clarification is None or bot._pending_clarification.done()


@pytest.mark.asyncio
async def test_unclaimed_non_start_prompts(tmp_path):
    tg = FakeTelegramClient()
    bot = _bot(tmp_path, admin_id=0, telegram=tg)
    await bot._handle_message(111, "hello")
    assert bot.admin_id == 0
    assert any("/start" in text for _, text in tg.sent)


@pytest.mark.asyncio
async def test_extract_handles_missing_text(tmp_path):
    bot = _bot(tmp_path)
    assert bot._extract({"update_id": 1}) is None
    parsed = bot._extract(
        {"update_id": 2, "message": {"text": "hi", "chat": {"id": 5}}}
    )
    assert parsed == (5, "hi")
