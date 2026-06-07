import pytest

from command_center.executor import ShellExecutor, _parse_fix
from command_center.notifier import CollectingNotifier
from tests.fakes import FakeModelClient


def _executor(tmp_path, model_client, max_heal_attempts=3):
    return ShellExecutor(
        workdir=str(tmp_path),
        timeout=30,
        max_heal_attempts=max_heal_attempts,
        model_client=model_client,
        fixer_model="fixer",
        notify=CollectingNotifier(),
    )


@pytest.mark.asyncio
async def test_successful_command(tmp_path):
    ex = _executor(tmp_path, FakeModelClient())
    result = await ex.run("echo hello")
    assert result.ok
    assert "hello" in result.stdout
    assert result.attempts == 1
    assert result.healed is False


@pytest.mark.asyncio
async def test_self_heal_recovers(tmp_path):
    fixer = FakeModelClient(['{"diagnosis": "use true", "new_command": "true"}'])
    ex = _executor(tmp_path, fixer)
    result = await ex.run("false")
    assert result.ok
    assert result.healed is True
    assert result.attempts == 2
    assert len(fixer.calls) == 1


@pytest.mark.asyncio
async def test_gives_up_when_no_fix(tmp_path):
    fixer = FakeModelClient(['{"diagnosis": "no idea", "new_command": ""}'])
    ex = _executor(tmp_path, fixer)
    result = await ex.run("false")
    assert not result.ok
    assert result.healed is False


@pytest.mark.asyncio
async def test_timeout(tmp_path):
    ex = ShellExecutor(
        workdir=str(tmp_path),
        timeout=1,
        max_heal_attempts=0,
        model_client=FakeModelClient(),
        fixer_model="fixer",
        notify=CollectingNotifier(),
    )
    result = await ex.run("sleep 5")
    assert result.returncode == 124


def test_write_file_inside_workspace(tmp_path):
    ex = _executor(tmp_path, FakeModelClient())
    target = ex.write_file("sub/dir/app.py", "print('hi')")
    assert target.read_text() == "print('hi')"


def test_write_file_rejects_escape(tmp_path):
    ex = _executor(tmp_path, FakeModelClient())
    with pytest.raises(ValueError):
        ex.write_file("../escape.py", "x")


def test_parse_fix_with_fences():
    raw = '```json\n{"diagnosis": "d", "new_command": "ls"}\n```'
    diagnosis, cmd = _parse_fix(raw)
    assert diagnosis == "d"
    assert cmd == "ls"


def test_parse_fix_garbage():
    diagnosis, cmd = _parse_fix("totally not json")
    assert cmd == ""
