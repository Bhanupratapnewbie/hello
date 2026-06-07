"""The Brain: intent analysis, decomposition, and specialist routing.

A request is decomposed by the planner model into ordered steps, each tagged with
the specialist role best suited to carry it out:

* ``shell``      -> the self-healing execution loop (the body)
* ``write_file`` -> the code-generation model produces file contents
* ``reason``     -> the long-context reasoning model analyses / researches
* ``clarify``    -> pause and ask the administrator via Telegram

A leading ``!`` makes the brain treat the whole message as a raw shell command and
hand it straight to the executor, bypassing planning entirely.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .config import Settings
from .executor import ShellExecutor
from .models import ChatMessage, ModelClient, ModelError
from .notifier import Notifier

AskAdmin = Callable[[str], Awaitable[str]]

_PLANNER_SYSTEM = (
    "You are the routing brain of an autonomous command center operated by a "
    "single trusted administrator on their own machine. Decompose the "
    "administrator's request into a concise ordered list of executable steps. "
    "Respond ONLY with a JSON object of the form "
    '{"steps": [ {"action": "...", "description": "...", ...}, ... ]}. '
    "Valid actions and their fields:\n"
    '- {"action": "shell", "description": str, "command": str}\n'
    '- {"action": "write_file", "description": str, "path": str, "instruction": str}\n'
    '- {"action": "reason", "description": str, "prompt": str}\n'
    '- {"action": "clarify", "description": str, "question": str}\n'
    "Use the minimum number of steps. Prefer a single shell step for simple "
    "commands. Use clarify only when a parameter is genuinely ambiguous and you "
    "cannot proceed."
)

_CODER_SYSTEM = (
    "You are a code-generation engine. Output ONLY the raw file contents for the "
    "requested file, with no markdown fences, commentary, or explanation."
)


@dataclass
class Step:
    action: str
    description: str
    command: str = ""
    path: str = ""
    instruction: str = ""
    prompt: str = ""
    question: str = ""


class Brain:
    def __init__(
        self,
        *,
        settings: Settings,
        model_client: ModelClient,
        executor: ShellExecutor,
        notify: Notifier,
        ask_admin: AskAdmin,
    ) -> None:
        self.settings = settings
        self.model_client = model_client
        self.executor = executor
        self.notify = notify
        self.ask_admin = ask_admin

    async def plan(self, request: str) -> list[Step]:
        messages = [
            ChatMessage("system", _PLANNER_SYSTEM),
            ChatMessage("user", request),
        ]
        raw = await self.model_client.complete(
            self.settings.role_model("planner"), messages, temperature=0.1
        )
        return parse_plan(raw)

    async def handle(self, request: str) -> None:
        request = request.strip()
        if not request:
            return

        # Raw, unfiltered shell passthrough: "!<command>".
        if request.startswith("!"):
            command = request[1:].strip()
            await self.notify("raw shell mode")
            await self.executor.run(command)
            await self.notify("done")
            return

        await self.notify("analysing request and building plan...")
        try:
            steps = await self.plan(request)
        except ModelError as exc:
            await self.notify(f"planner model error: {exc}")
            return

        if not steps:
            await self.notify("planner produced no steps; nothing to do")
            return

        plan_view = "\n".join(
            f"{i + 1}. [{s.action}] {s.description}" for i, s in enumerate(steps)
        )
        await self.notify(f"plan ({len(steps)} steps):\n{plan_view}")

        context: list[str] = []
        for i, step in enumerate(steps, start=1):
            await self.notify(f"step {i}/{len(steps)}: {step.description}")
            try:
                outcome = await self._run_step(step, context)
            except ModelError as exc:
                await self.notify(f"model error on step {i}: {exc}")
                return
            if outcome is not None:
                context.append(outcome)

        await self.notify("all steps complete")

    async def _run_step(self, step: Step, context: list[str]) -> str | None:
        if step.action == "shell":
            result = await self.executor.run(step.command)
            return f"shell `{step.command}` -> exit {result.returncode}"
        if step.action == "write_file":
            content = await self._generate_file(step, context)
            target = self.executor.write_file(step.path, content)
            await self.notify(f"wrote {target} ({len(content)} bytes)")
            return f"wrote file {step.path}"
        if step.action == "reason":
            analysis = await self._reason(step, context)
            await self.notify(analysis)
            return analysis
        if step.action == "clarify":
            answer = await self.ask_admin(step.question)
            return f"clarification: {step.question} -> {answer}"
        await self.notify(f"unknown action '{step.action}'; skipping")
        return None

    async def _generate_file(self, step: Step, context: list[str]) -> str:
        ctx = ("\n".join(context))[-4000:]
        user = (
            f"File path: {step.path}\n"
            f"Requirements: {step.instruction}\n"
            f"Prior context:\n{ctx}\n"
        )
        return await self.model_client.complete(
            self.settings.role_model("coder"),
            [ChatMessage("system", _CODER_SYSTEM), ChatMessage("user", user)],
            temperature=0.1,
        )

    async def _reason(self, step: Step, context: list[str]) -> str:
        ctx = ("\n".join(context))[-6000:]
        user = f"{step.prompt}\n\nPrior context:\n{ctx}"
        return await self.model_client.complete(
            self.settings.role_model("reasoner"),
            [ChatMessage("user", user)],
            temperature=0.3,
        )


def parse_plan(raw: str) -> list[Step]:
    """Parse a planner response into a list of Steps, tolerating markdown noise."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    raw_steps = data.get("steps", [])
    steps: list[Step] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        steps.append(
            Step(
                action=action,
                description=str(item.get("description", "")).strip(),
                command=str(item.get("command", "")).strip(),
                path=str(item.get("path", "")).strip(),
                instruction=str(item.get("instruction", "")).strip(),
                prompt=str(item.get("prompt", "")).strip(),
                question=str(item.get("question", "")).strip(),
            )
        )
    return steps
