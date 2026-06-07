# Command Center

An autonomous agentic command center: a multi-model **routing brain** wired to a
**self-healing shell/filesystem execution loop**, controlled exclusively through
**Telegram**.

It plans a request, routes each sub-task to the model best suited for it, executes
shell/file work, captures failures, asks a fast model to diagnose and correct them,
retries until done, and streams every step back to the administrator on Telegram.

## Architecture

### 1. The Brain — intent analysis & specialist routing (`brain.py`)
Decomposes an administrator request into ordered steps and routes each to a role:

| Role       | Used for                                             | Env var          |
|------------|------------------------------------------------------|------------------|
| `planner`  | Macro blueprint + routing                            | `MODEL_PLANNER`  |
| `reasoner` | Deep, long-context reasoning, docs analysis, research| `MODEL_REASONER` |
| `coder`    | Pure code/file generation across languages           | `MODEL_CODER`    |
| `fixer`    | Low-latency error parsing & corrective adjustments   | `MODEL_FIXER`    |

The model layer is **provider-agnostic** (OpenAI-compatible Chat Completions). Point
every role at any endpoint/model you want — OpenRouter, OpenAI, Together, a local
vLLM or **Ollama** instance, etc. — via `MODEL_BASE_URL` + `MODEL_API_KEY`.

### 2. The Body — self-healing execution environment (`executor.py`)
Runs shell commands in a workspace, captures stdout/stderr natively, and on failure
sends the error to the `fixer` model, applies the suggested corrected command, and
retries up to `CC_MAX_HEAL_ATTEMPTS` times. Can also write generated files.

### 3. The Gateway — Telegram (`bot.py`, `telegram.py`)
The **sole** control/communication interface.
- Long-polls Telegram (dependency-light, just `httpx`).
- **Single-administrator** access: the first person to send `/start` is locked in as
  admin and persisted; everyone else is ignored.
- Streams logs, plans, command output, and final results to the admin.
- Pauses and asks the admin for **clarification** on ambiguous steps, then resumes.

The optional FastAPI wrapper (`app.py`) exposes only `/` and `/health` for
deployment — never a control surface — and launches the bot as a background task.

## Usage

| Input                         | Behaviour                                            |
|-------------------------------|------------------------------------------------------|
| plain text task               | plan → route → execute → self-heal → report          |
| `!<command>`                  | run as a raw shell command immediately (passthrough) |
| `/start`                      | claim admin (first time) / show help                 |
| `/status`                     | show admin binding + busy state                      |
| `/cancel`                     | abort the current running task                        |

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN and a model key (or local Ollama)
python -m command_center
```

`TELEGRAM_ADMIN_CHAT_ID` is optional — leave it `0` and the bot auto-claims the
first `/start` sender.

### Running fully local / uncensored (no external key)

```bash
ollama serve &
ollama pull dolphin-mistral
# in .env:
#   MODEL_BASE_URL=http://localhost:11434/v1
#   MODEL_API_KEY=ollama
#   MODEL_PLANNER=dolphin-mistral   (and the other roles)
```

### Deploy (ASGI)

```bash
uvicorn command_center.app:app --host 0.0.0.0 --port 8000
```

## Development

```bash
ruff check .
pytest -q
```

## Configuration reference

See [`.env.example`](.env.example) for every variable.
