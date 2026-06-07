"""The Body: a self-healing shell + filesystem execution environment.

Runs shell commands inside a working directory, captures stdout/stderr natively,
and on failure routes the error through the fast "fixer" model to obtain a
corrected command, retrying until success or until the retry budget is spent.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .models import ChatMessage, ModelClient, ModelError
from .notifier import Notifier

_FIXER_SYSTEM = (
    "You are a fast error-parsing utility for an autonomous command runner. "
    "You are given a shell command that failed and its captured stderr/stdout. "
    "Diagnose the root cause and, when possible, output a corrected shell command "
    "that should be run instead. Respond ONLY with a compact JSON object of the "
    'form {"diagnosis": "...", "new_command": "..."}. If no safe automatic fix '
    'exists, set "new_command" to an empty string.'
)


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    attempts: int = 1
    healed: bool = False
    history: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def summary(self, max_len: int = 1500) -> str:
        status = "OK" if self.ok else f"FAILED (exit {self.returncode})"
        body = self.stdout if self.ok else (self.stderr or self.stdout)
        body = body.strip()
        if len(body) > max_len:
            body = body[:max_len] + "\n... [truncated]"
        prefix = f"$ {self.command}\n[{status}]"
        return f"{prefix}\n{body}" if body else prefix


class ShellExecutor:
    def __init__(
        self,
        *,
        workdir: str,
        timeout: int,
        max_heal_attempts: int,
        model_client: ModelClient,
        fixer_model: str,
        notify: Notifier,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.timeout = timeout
        self.max_heal_attempts = max_heal_attempts
        self.model_client = model_client
        self.fixer_model = fixer_model
        self.notify = notify
        self.workdir.mkdir(parents=True, exist_ok=True)

    async def _run_once(self, command: str) -> CommandResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workdir),
                env=os.environ.copy(),
            )
        except OSError as exc:
            return CommandResult(command, 127, "", f"failed to spawn process: {exc}")

        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return CommandResult(
                command, 124, "", f"command timed out after {self.timeout}s"
            )

        return CommandResult(
            command=command,
            returncode=proc.returncode if proc.returncode is not None else 1,
            stdout=out_b.decode(errors="replace"),
            stderr=err_b.decode(errors="replace"),
        )

    async def _diagnose(self, result: CommandResult) -> tuple[str, str]:
        """Ask the fixer model for a diagnosis and a corrected command."""
        prompt = (
            f"Command:\n{result.command}\n\n"
            f"Exit code: {result.returncode}\n\n"
            f"stderr:\n{result.stderr[-3000:]}\n\n"
            f"stdout:\n{result.stdout[-1000:]}\n"
        )
        messages = [
            ChatMessage("system", _FIXER_SYSTEM),
            ChatMessage("user", prompt),
        ]
        try:
            raw = await self.model_client.complete(
                self.fixer_model, messages, temperature=0.0
            )
        except ModelError as exc:
            return (f"fixer model unavailable: {exc}", "")

        diagnosis, new_command = _parse_fix(raw)
        return diagnosis, new_command

    async def run(self, command: str) -> CommandResult:
        """Run a command, self-healing on failure up to the retry budget."""
        history: list[str] = []
        current = command
        attempts = 0
        healed = False

        while True:
            attempts += 1
            await self.notify(f"running: `{current}`")
            result = await self._run_once(current)
            history.append(current)
            await self.notify(result.summary())

            if result.ok or attempts > self.max_heal_attempts:
                result.attempts = attempts
                result.healed = healed
                result.history = history
                return result

            await self.notify(
                f"command failed (exit {result.returncode}); diagnosing "
                f"[attempt {attempts}/{self.max_heal_attempts}]"
            )
            diagnosis, new_command = await self._diagnose(result)
            await self.notify(f"diagnosis: {diagnosis}")

            if not new_command or new_command.strip() == current.strip():
                await self.notify("no automatic fix available; giving up on this step")
                result.attempts = attempts
                result.healed = healed
                result.history = history
                return result

            healed = True
            current = new_command

    def write_file(self, relative_path: str, content: str) -> Path:
        """Write a file inside the workspace, creating parent dirs as needed."""
        target = (self.workdir / relative_path).resolve()
        if not str(target).startswith(str(self.workdir)):
            raise ValueError(f"refusing to write outside workspace: {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target


def _parse_fix(raw: str) -> tuple[str, str]:
    """Extract (diagnosis, new_command) from a model response, tolerating noise."""
    text = raw.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            return (
                str(data.get("diagnosis", "")).strip(),
                str(data.get("new_command", "")).strip(),
            )
        except json.JSONDecodeError:
            pass
    # Fallback: treat the whole thing as a diagnosis, no command.
    return (text[:500], "")


def safe_display_command(command: str) -> str:
    """Best-effort single-line rendering of a command for logs."""
    try:
        return " ".join(shlex.split(command))
    except ValueError:
        return command.replace("\n", " ")
