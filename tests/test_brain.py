import pytest

from command_center.brain import (
    Brain,
    Step,
    SupervisorVerdict,
    _is_looping,
    _step_signature,
    parse_plan,
    parse_verdict,
)
from command_center.config import Settings
from command_center.executor import ShellExecutor
from command_center.notifier import CollectingNotifier
from tests.fakes import FakeModelClient


def _settings(tmp_path, **overrides):
    return Settings(
        telegram_bot_token="t",
        telegram_admin_chat_id=1,
        model_api_key="k",
        workdir=str(tmp_path),
        **overrides,
    )


def _brain(tmp_path, model_client, notifier, ask_answers=None, settings=None):
    answers = list(ask_answers or [])

    async def ask_admin(question: str) -> str:
        return answers.pop(0) if answers else ""

    settings = settings or _settings(tmp_path)
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


def test_parse_plan_salvages_truncated_json():
    # Two complete steps then a third cut off mid-string (truncation), with no
    # closing braces for the array/object — the complete two must be recovered.
    raw = (
        '{"steps": ['
        '{"action": "shell", "description": "a", "command": "mkdir site"}, '
        '{"action": "write_file", "description": "b", "path": "site/index.html", '
        '"instruction": "landing page"}, '
        '{"action": "shell", "description": "c", "command": "echo <!DOCTYPE html'
    )
    steps = parse_plan(raw)
    assert [s.action for s in steps] == ["shell", "write_file"]
    assert steps[1].path == "site/index.html"


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
async def test_clarify_only_plan_replans_with_answer(tmp_path):
    # A clarify-only plan must not end the task: the answer is collected and the
    # request re-planned into real work.
    clarify = (
        '{"steps": [{"action": "clarify", "description": "ask", '
        '"question": "which port?"}]}'
    )
    real = (
        '{"steps": [{"action": "shell", "description": "deploy", '
        '"command": "echo deploying"}]}'
    )
    model = FakeModelClient([clarify, real])
    notifier = CollectingNotifier()
    brain = _brain(tmp_path, model, notifier, ask_answers=["8080"])
    await brain.handle("deploy something")
    assert "deploying" in notifier.transcript
    assert "all steps complete" in notifier.transcript
    # The admin's answer was folded into the re-plan request.
    replan_messages = model.calls[-1][1]
    assert any("8080" in m.content for m in replan_messages)


@pytest.mark.asyncio
async def test_plan_retries_until_valid(tmp_path):
    # First reply is unusable prose; the planner is nudged and the retry yields a
    # valid plan instead of giving up with "nothing to do".
    good = (
        '{"steps": [{"action": "shell", "description": "hi", '
        '"command": "echo hi"}]}'
    )
    model = FakeModelClient(["sorry, here is what I think...", good])
    notifier = CollectingNotifier()
    brain = _brain(tmp_path, model, notifier)
    await brain.handle("do the thing")
    assert "hi" in notifier.transcript
    assert "all steps complete" in notifier.transcript
    assert "nothing to do" not in notifier.transcript


@pytest.mark.asyncio
async def test_plan_gives_up_after_attempts(tmp_path):
    model = FakeModelClient(["nope", "still nope", "nope again"])
    notifier = CollectingNotifier()
    settings = _settings(tmp_path, plan_attempts=3)
    brain = _brain(tmp_path, model, notifier, settings=settings)
    await brain.handle("impossible to parse")
    assert "planner produced no steps; nothing to do" in notifier.transcript
    # It actually tried the configured number of times.
    assert len(model.calls) == 3


# --- Supervisor (watchdog) ---------------------------------------------------


def test_parse_verdict_variants():
    assert parse_verdict('{"verdict": "abort", "reason": "done"}').verdict == "abort"
    v = parse_verdict('```json\n{"verdict":"change","reason":"loop"}\n```')
    assert v.verdict == "change" and v.reason == "loop"
    # Unknown / unparseable defaults to a safe "continue".
    assert parse_verdict("garbage").verdict == "continue"
    assert parse_verdict('{"verdict": "weird"}').verdict == "continue"


def test_step_signature_and_loop_detection():
    s = Step(action="shell", description="d", command="echo x")
    assert _step_signature(s) == "shell:echo x"
    sigs = ["shell:echo x"] * 3
    assert _is_looping(sigs, 3) is True
    assert _is_looping(["shell:echo x", "shell:echo y", "shell:echo x"], 3) is False
    # A bare action with no payload is too weak to call a loop.
    assert _is_looping(["reason:", "reason:", "reason:"], 3) is False


def _repeated_plan(command: str, n: int) -> str:
    steps = ", ".join(
        f'{{"action": "shell", "description": "loop {i}", "command": "{command}"}}'
        for i in range(n)
    )
    return f'{{"steps": [{steps}]}}'


@pytest.mark.asyncio
async def test_supervisor_detects_loop_and_replans(tmp_path):
    plan = _repeated_plan("echo loop", 4)
    change = '{"verdict": "change", "reason": "stuck repeating echo loop"}'
    fixed = '{"steps": [{"action": "shell", "description": "fix", "command": "echo fixed"}]}'
    model = FakeModelClient([plan, change, fixed])
    notifier = CollectingNotifier()
    settings = _settings(tmp_path, supervisor_interval=0, loop_threshold=3)
    brain = _brain(tmp_path, model, notifier, settings=settings)
    await brain.handle("keep echoing")
    assert "same action repeated" in notifier.transcript
    assert "supervisor verdict: change" in notifier.transcript
    assert "supervisor re-planned" in notifier.transcript
    assert "fixed" in notifier.transcript
    assert "all steps complete" in notifier.transcript


@pytest.mark.asyncio
async def test_supervisor_abort_stops_task(tmp_path):
    plan = _repeated_plan("echo loop", 4)
    abort = '{"verdict": "abort", "reason": "goal unreachable"}'
    model = FakeModelClient([plan, abort])
    notifier = CollectingNotifier()
    settings = _settings(tmp_path, supervisor_interval=0, loop_threshold=3)
    brain = _brain(tmp_path, model, notifier, settings=settings)
    await brain.handle("keep echoing")
    assert "supervisor: aborting task" in notifier.transcript
    assert "all steps complete" not in notifier.transcript


@pytest.mark.asyncio
async def test_supervisor_asks_admin(tmp_path):
    plan = _repeated_plan("echo loop", 4)
    ask = '{"verdict": "ask_admin", "question": "which framework?"}'
    model = FakeModelClient([plan, ask])
    notifier = CollectingNotifier()
    settings = _settings(tmp_path, supervisor_interval=0, loop_threshold=3)
    brain = _brain(
        tmp_path, model, notifier, ask_answers=["flask"], settings=settings
    )
    await brain.handle("keep echoing")
    assert "supervisor verdict: ask_admin" in notifier.transcript
    assert "all steps complete" in notifier.transcript


@pytest.mark.asyncio
async def test_step_ceiling_stops_runaway(tmp_path):
    plan = (
        '{"steps": ['
        '{"action": "shell", "description": "a", "command": "echo a"},'
        '{"action": "shell", "description": "b", "command": "echo b"},'
        '{"action": "shell", "description": "c", "command": "echo c"}'
        "]}"
    )
    model = FakeModelClient([plan])
    notifier = CollectingNotifier()
    settings = _settings(
        tmp_path, max_total_steps=2, supervisor_interval=0, loop_threshold=99
    )
    brain = _brain(tmp_path, model, notifier, settings=settings)
    await brain.handle("do three things")
    assert "hit the step ceiling" in notifier.transcript
    assert "all steps complete" not in notifier.transcript


@pytest.mark.asyncio
async def test_supervisor_disabled_runs_plainly(tmp_path):
    plan = _repeated_plan("echo loop", 4)
    model = FakeModelClient([plan])
    notifier = CollectingNotifier()
    settings = _settings(tmp_path, supervisor_enabled=False)
    brain = _brain(tmp_path, model, notifier, settings=settings)
    await brain.handle("keep echoing")
    # No supervisor activity, and only the plan call is made (no verdict calls).
    assert "supervisor" not in notifier.transcript
    assert "all steps complete" in notifier.transcript
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_supervisor_continue_does_not_replan(tmp_path):
    plan = _repeated_plan("echo loop", 4)
    cont = '{"verdict": "continue", "reason": "looks fine"}'
    model = FakeModelClient([plan, cont])
    notifier = CollectingNotifier()
    settings = _settings(tmp_path, supervisor_interval=0, loop_threshold=3)
    brain = _brain(tmp_path, model, notifier, settings=settings)
    await brain.handle("keep echoing")
    assert "supervisor verdict: continue" in notifier.transcript
    assert "supervisor re-planned" not in notifier.transcript
    assert "all steps complete" in notifier.transcript


def test_supervisor_verdict_dataclass_defaults():
    v = SupervisorVerdict("continue")
    assert v.reason == "" and v.question == ""
