import asyncio

import pytest

from command_center.bot import CommandCenterBot
from command_center.config import Settings
from tests.fakes import FakeModelClient, FakeTelegramClient


def _settings(tmp_path, admin_id=0, claim_token=""):
    return Settings(
        telegram_bot_token="t",
        telegram_admin_chat_id=admin_id,
        admin_claim_token=claim_token,
        model_api_key="k",
        workdir=str(tmp_path),
    )


def _bot(tmp_path, admin_id=0, telegram=None, claim_token=""):
    return CommandCenterBot(
        _settings(tmp_path, admin_id, claim_token),
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
async def test_second_admin_joins_with_token(tmp_path):
    tg = FakeTelegramClient()
    bot = _bot(tmp_path, admin_id=111, telegram=tg, claim_token="s3cret")
    await bot._handle_message(222, "/start s3cret")
    assert bot.admin_ids == [111, 222]
    # Both admins are persisted, and the new admin gets the help text.
    saved = (tmp_path / ".cc_admin").read_text().split()
    assert saved == ["111", "222"]
    assert any(chat == 222 for chat, _ in tg.sent)


@pytest.mark.asyncio
async def test_wrong_token_does_not_grant_admin(tmp_path):
    tg = FakeTelegramClient()
    bot = _bot(tmp_path, admin_id=111, telegram=tg, claim_token="s3cret")
    await bot._handle_message(222, "/start nope")
    assert bot.admin_ids == [111]
    assert all("online" not in text for _, text in tg.sent)


@pytest.mark.asyncio
async def test_both_admins_receive_notifications(tmp_path):
    tg = FakeTelegramClient()
    bot = _bot(tmp_path, admin_id=111, telegram=tg, claim_token="s3cret")
    await bot._handle_message(222, "/start s3cret")
    tg.sent.clear()
    await bot._notify("hello admins")
    targets = {chat for chat, _ in tg.sent}
    assert targets == {111, 222}


def test_load_admin_ids_legacy_single_line(tmp_path):
    (tmp_path / ".cc_admin").write_text("6998977616")
    bot = _bot(tmp_path, admin_id=0)
    assert bot.admin_ids == [6998977616]
    assert bot.admin_id == 6998977616


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
