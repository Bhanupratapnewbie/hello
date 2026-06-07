import pytest

from command_center.brain import Brain, parse_plan
from command_center.config import Settings
from command_center.executor import ShellExecutor
from command_center.notifier import CollectingNotifier
from tests.fakes import FakeModelClient


def _settings(tmp_path):
    return Settings(
        telegram_bot_token="t",
        telegram_admin_chat_id=1,
        model_api_key="k",
        workdir=str(tmp_path),
    )


def _brain(tmp_path, model_client, notifier, ask_answers=None):
    answers = list(ask_answers or [])

    async def ask_admin(question: str) -> str:
        return answers.pop(0) if answers else ""

    settings = _settings(tmp_path)
    executor = ShellExecutor(
        workdir=str(tmp_path),
        timeout=30,
        max_heal_attempts=2,
        model_client=model_client,
        fixer_model="fixer",
        notify=notifier,
    )
    return Brain(
        settings=settings,
        model_client=model_client,
        executor=executor,
        notify=notifier,
        ask_admin=ask_admin,
    )


def test_parse_plan_with_fences():
    raw = '```json\n{"steps": [{"action": "shell", "description": "d", "command": "ls"}]}\n```'
    steps = parse_plan(raw)
    assert len(steps) == 1
    assert steps[0].action == "shell"
    assert steps[0].command == "ls"


def test_parse_plan_invalid():
    assert parse_plan("no json here") == []


@pytest.mark.asyncio
async def test_raw_shell_passthrough(tmp_path):
    notifier = CollectingNotifier()
    brain = _brain(tmp_path, FakeModelClient(), notifier)
    await brain.handle("!echo passthrough")
    assert "passthrough" in notifier.transcript
    # No model calls in raw mode.
    assert brain.model_client.calls == []


@pytest.mark.asyncio
async def test_planned_shell_step(tmp_path):
    plan = '{"steps": [{"action": "shell", "description": "say hi", "command": "echo hi"}]}'
    model = FakeModelClient([plan])
    notifier = CollectingNotifier()
    brain = _brain(tmp_path, model, notifier)
    await brain.handle("say hi in the shell")
    assert "hi" in notifier.transcript
    assert "all steps complete" in notifier.transcript


@pytest.mark.asyncio
async def test_write_file_step(tmp_path):
    plan = (
        '{"steps": [{"action": "write_file", "description": "make app", '
        '"path": "app.py", "instruction": "hello world"}]}'
    )
    model = FakeModelClient([plan, "print('hello world')"])
    notifier = CollectingNotifier()
    brain = _brain(tmp_path, model, notifier)
    await brain.handle("write an app")
    written = (tmp_path / "app.py").read_text()
    assert "hello world" in written


@pytest.mark.asyncio
async def test_clarify_step(tmp_path):
    plan = '{"steps": [{"action": "clarify", "description": "ask", "question": "which port?"}]}'
    model = FakeModelClient([plan])
    notifier = CollectingNotifier()
    brain = _brain(tmp_path, model, notifier, ask_answers=["8080"])
    await brain.handle("deploy something")
    assert "all steps complete" in notifier.transcript
