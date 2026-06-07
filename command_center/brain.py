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
    "cannot proceed.\n"
    "IMPORTANT: To create or overwrite a file, ALWAYS use the write_file action "
    "with a short natural-language 'instruction' describing the file's contents. "
    "NEVER embed file bodies inside a shell command (no heredocs, no 'cat > file "
    "<< EOF', no large 'echo'). Keep every step's JSON small so the plan stays "
    "valid — the code-generation model fills in the actual file contents later."
)

_CODER_SYSTEM = (
    "You are a code-generation engine. Output ONLY the raw file contents for the "
    "requested file, with no markdown fences, commentary, or explanation."
)

_SUPERVISOR_SYSTEM = (
    "You are the progress supervisor of an autonomous command center. You watch a "
    "task that is already in flight and judge whether it is still on the right "
    "path toward the administrator's goal, or whether it is stuck/looping/drifting. "
    "You are given the original goal, the work done so far, and the steps still "
    "queued. Respond ONLY with a JSON object: "
    '{"verdict": "continue|change|ask_admin|abort", "reason": str, '
    '"question": str}. '
    "Use 'continue' if progress is healthy. Use 'change' if the remaining plan is "
    "wrong, redundant, or looping and should be re-planned. Use 'ask_admin' if you "
    "genuinely need a decision/credential/info from the administrator (put it in "
    "'question'). Use 'abort' only if the goal is impossible or already met. Be "
    "decisive and brief."
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


@dataclass
class SupervisorVerdict:
    verdict: str
    reason: str = ""
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

    async def _complete_role(
        self,
        role: str,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int | None = None,
    ) -> str:
        """Run a completion for a role on that role's configured provider."""
        base_url, api_key = self.settings.role_endpoint(role)
        return await self.model_client.complete(
            self.settings.role_model(role),
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    async def plan(self, request: str) -> list[Step]:
        """Build a plan, re-asking the model when it returns no usable steps.

        The planner sometimes replies with prose or truncated/empty JSON; instead
        of silently giving up we nudge it (up to ``plan_attempts`` times) to emit a
        valid, non-empty JSON plan.
        """
        messages = [
            ChatMessage("system", _PLANNER_SYSTEM),
            ChatMessage("user", request),
        ]
        attempts = max(1, self.settings.plan_attempts)
        for attempt in range(attempts):
            raw = await self._complete_role(
                "planner",
                messages,
                temperature=0.1,
                max_tokens=self.settings.plan_max_tokens,
            )
            steps = parse_plan(raw)
            if steps:
                return steps
            if attempt < attempts - 1:
                messages.append(ChatMessage("assistant", raw[:1000]))
                messages.append(
                    ChatMessage(
                        "user",
                        "That was not usable. Respond with ONLY a JSON object "
                        '{"steps": [...]} containing at least one concrete, '
                        "executable step (shell / write_file / reason). Do not "
                        "return prose, an empty list, or only a clarify step "
                        "unless a parameter is truly unknowable.",
                    )
                )
        return []

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
            # If the planner only wants clarification, answer it and re-plan so a
            # task isn't ended just because a question was asked.
            rounds = 0
            while (
                steps
                and all(s.action == "clarify" for s in steps)
                and rounds < self.settings.max_clarify_rounds
            ):
                answers = []
                for s in steps:
                    answer = await self.ask_admin(s.question)
                    answers.append(f"- {s.question} -> {answer}")
                request = (
                    f"{request}\n\nAdditional details provided by the "
                    f"administrator:\n" + "\n".join(answers)
                )
                rounds += 1
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

        await self._execute_plan(request, steps)

    async def _execute_plan(self, request: str, steps: list[Step]) -> None:
        """Run a plan under the progress supervisor (watchdog).

        The supervisor guards the whole task end-to-end: it caps the total number
        of executed steps, detects when the same action repeats (a loop), and
        periodically asks the base model to judge whether the task is still on the
        right path — continuing, re-planning the rest, asking the admin, or
        aborting. With the supervisor disabled it degrades to a plain run.
        """
        sup = self.settings.supervisor_enabled
        context: list[str] = []
        pending: list[Step] = list(steps)
        recent_sigs: list[str] = []
        executed = 0
        replans = 0

        while pending:
            if sup and executed >= self.settings.max_total_steps:
                await self.notify(
                    "supervisor: hit the step ceiling "
                    f"({self.settings.max_total_steps}) — stopping to avoid a "
                    "runaway loop"
                )
                return

            step = pending.pop(0)
            executed += 1
            await self.notify(f"step {executed}: {step.description}")
            try:
                outcome = await self._run_step(step, context)
            except ModelError as exc:
                await self.notify(f"model error on step {executed}: {exc}")
                return
            if outcome is not None:
                context.append(outcome)

            if not sup:
                continue

            recent_sigs.append(_step_signature(step))
            looping = _is_looping(recent_sigs, self.settings.loop_threshold)
            interval = self.settings.supervisor_interval
            due = interval > 0 and executed % interval == 0

            if pending and (looping or due):
                if looping:
                    await self.notify(
                        "supervisor: same action repeated "
                        f"{self.settings.loop_threshold}x — checking course"
                    )
                verdict = await self._supervise(request, context, pending)
                await self.notify(
                    f"supervisor verdict: {verdict.verdict} — {verdict.reason}"
                )
                if verdict.verdict == "abort":
                    await self.notify("supervisor: aborting task")
                    return
                if verdict.verdict == "ask_admin":
                    question = verdict.question or verdict.reason or (
                        "I need your input to proceed — how should I continue?"
                    )
                    answer = await self.ask_admin(question)
                    context.append(f"admin guidance: {question} -> {answer}")
                    recent_sigs.clear()
                elif verdict.verdict == "change":
                    if replans >= self.settings.max_replans:
                        await self.notify(
                            "supervisor: re-plan limit reached — asking admin"
                        )
                        answer = await self.ask_admin(
                            "I'm stuck re-planning this task. How should I "
                            "proceed?"
                        )
                        context.append(f"admin guidance -> {answer}")
                        recent_sigs.clear()
                    else:
                        replans += 1
                        new_steps = await self._replan(
                            request, context, pending, verdict.reason
                        )
                        if new_steps:
                            pending = new_steps
                            recent_sigs.clear()
                            new_view = "\n".join(
                                f"{j + 1}. [{s.action}] {s.description}"
                                for j, s in enumerate(pending)
                            )
                            await self.notify(
                                f"supervisor re-planned ({len(pending)} "
                                f"steps):\n{new_view}"
                            )

        await self.notify("all steps complete")

    async def _supervise(
        self, request: str, context: list[str], pending: list[Step]
    ) -> SupervisorVerdict:
        """Ask the base model to judge whether the task is on the right path."""
        done = ("\n".join(context))[-4000:]
        queued = "\n".join(f"- [{s.action}] {s.description}" for s in pending)
        user = (
            f"Original goal:\n{request}\n\n"
            f"Work done so far:\n{done or '(nothing yet)'}\n\n"
            f"Steps still queued:\n{queued or '(none)'}\n\n"
            "Is this on the right path? Respond with the JSON verdict."
        )
        try:
            raw = await self._complete_role(
                "planner",
                [
                    ChatMessage("system", _SUPERVISOR_SYSTEM),
                    ChatMessage("user", user),
                ],
                temperature=0.0,
            )
        except ModelError as exc:
            # Supervisor must never crash the task; default to continuing.
            await self.notify(f"supervisor check failed ({exc}); continuing")
            return SupervisorVerdict("continue", "supervisor unavailable")
        return parse_verdict(raw)

    async def _replan(
        self,
        request: str,
        context: list[str],
        pending: list[Step],
        reason: str,
    ) -> list[Step]:
        """Re-plan the remaining work after the supervisor flags a problem."""
        done = ("\n".join(context))[-4000:]
        queued = "\n".join(f"- [{s.action}] {s.description}" for s in pending)
        user = (
            f"Original goal:\n{request}\n\n"
            f"Work already done:\n{done or '(nothing yet)'}\n\n"
            f"The current remaining plan was flagged as off-track: {reason}\n"
            f"Current remaining steps:\n{queued or '(none)'}\n\n"
            "Produce a corrected plan for ONLY the remaining work to reach the "
            "goal, avoiding the repeated/ineffective actions above."
        )
        try:
            raw = await self._complete_role(
                "planner",
                [
                    ChatMessage("system", _PLANNER_SYSTEM),
                    ChatMessage("user", user),
                ],
                temperature=0.1,
            )
        except ModelError as exc:
            await self.notify(f"supervisor re-plan failed ({exc}); keeping plan")
            return pending
        return parse_plan(raw)

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
        return await self._complete_role(
            "coder",
            [ChatMessage("system", _CODER_SYSTEM), ChatMessage("user", user)],
            temperature=0.1,
        )

    async def _reason(self, step: Step, context: list[str]) -> str:
        ctx = ("\n".join(context))[-6000:]
        user = f"{step.prompt}\n\nPrior context:\n{ctx}"
        return await self._complete_role(
            "reasoner", [ChatMessage("user", user)], temperature=0.3
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
    raw_steps: list[object] = []
    if start != -1 and end != -1 and end > start:
        try:
            raw_steps = json.loads(text[start : end + 1]).get("steps", [])
        except (json.JSONDecodeError, AttributeError):
            raw_steps = []
    if not raw_steps:
        # The response was likely truncated mid-JSON (e.g. a huge inline file
        # body). Salvage every complete step object that did come through.
        raw_steps = _salvage_step_objects(text)

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


def _salvage_step_objects(text: str) -> list[dict]:
    """Recover complete ``{...}`` step objects from truncated/invalid JSON.

    Walks the text respecting string escaping, and json-parses each balanced
    top-level object that appears after the ``"steps"`` array opens, skipping any
    final object cut off by truncation.
    """
    anchor = text.find('"steps"')
    scan = text[anchor:] if anchor != -1 else text
    objects: list[dict] = []
    depth = 0
    in_str = False
    escaped = False
    obj_start = -1
    for i, ch in enumerate(scan):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start != -1:
                    try:
                        parsed = json.loads(scan[obj_start : i + 1])
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict) and "action" in parsed:
                        objects.append(parsed)
                    obj_start = -1
    return objects


def _step_signature(step: Step) -> str:
    """A stable identity for a step, used to detect repeating actions."""
    payload = step.command or step.path or step.prompt or step.question
    return f"{step.action}:{payload.strip()}"


def _is_looping(signatures: list[str], threshold: int) -> bool:
    """True when the last ``threshold`` signatures are identical and non-empty."""
    if threshold <= 1 or len(signatures) < threshold:
        return False
    window = signatures[-threshold:]
    last = window[0]
    # An empty payload (e.g. a bare description) is too weak to call a loop.
    if not last or ":" not in last or not last.split(":", 1)[1]:
        return False
    return all(sig == last for sig in window)


_VALID_VERDICTS = {"continue", "change", "ask_admin", "abort"}


def parse_verdict(raw: str) -> SupervisorVerdict:
    """Parse a supervisor response, defaulting to 'continue' on any ambiguity."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return SupervisorVerdict("continue", "unparseable verdict")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return SupervisorVerdict("continue", "unparseable verdict")
    if not isinstance(data, dict):
        return SupervisorVerdict("continue", "unparseable verdict")
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "continue"
    return SupervisorVerdict(
        verdict=verdict,
        reason=str(data.get("reason", "")).strip(),
        question=str(data.get("question", "")).strip(),
    )
