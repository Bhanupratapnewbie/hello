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
vLLM or **Ollama** instance, etc. — via the global `MODEL_BASE_URL` + `MODEL_API_KEY`,
and optionally **override the endpoint + key per role** with
`MODEL_<ROLE>_BASE_URL` / `MODEL_<ROLE>_API_KEY` (each falls back to the global one).

**Recommended default (the base is open weight, specialists are frontier):**
- `planner` — the **base/brain** every query hits first: the latest-class, fully
  **uncensored** open-weight **Qwen3** (abliterated; `Qwen3-32B` is the largest dense
  model — the only bigger Qwen3 is the 235B MoE), served **full-precision (no
  quantization)** on a cloud GPU box (vLLM or Ollama). Any latest uncensored
  Qwen/Gemma is a drop-in swap.
- `reasoner` — deep research → **Gemini 2.5 Pro** (`google/gemini-2.5-pro`).
- `coder` — code generation → **Claude Opus 4.1** (`anthropic/claude-opus-4.1`).
- `fixer` — fast error correction → **Gemini 2.5 Flash** (`google/gemini-2.5-flash`).

The per-role overrides keep the uncensored base on your **local GPU** while the
research/code/fix roles call cloud APIs — typically all fronted by a single
**OpenRouter** key. Set `MODEL_<ROLE>_BASE_URL=https://openrouter.ai/api/v1` and
`MODEL_<ROLE>_API_KEY=sk-or-...` for `reasoner`/`coder`/`fixer`; leave `planner`
with no override so it uses the local GPU endpoint.

### 2. The Body — self-healing execution environment (`executor.py`)
Runs shell commands in a workspace, captures stdout/stderr natively, and on failure
sends the error to the `fixer` model, applies the suggested corrected command, and
retries up to `CC_MAX_HEAL_ATTEMPTS` times. Can also write generated files.

### 2b. The Supervisor — progress watchdog (`brain.py`)
Wraps the whole task end-to-end so a long job can't silently spin or drift:
- **Step ceiling** — a hard cap (`CC_MAX_TOTAL_STEPS`) on total executed steps.
- **Loop detection** — if the same action repeats `CC_LOOP_THRESHOLD` times in a
  row, it breaks out and re-evaluates instead of grinding forever.
- **On-track checks** — every `CC_SUPERVISOR_INTERVAL` steps (and whenever a loop
  is detected) the **uncensored base** judges the run and returns a verdict:
  `continue`, `change` (re-plan the remaining work, up to `CC_MAX_REPLANS`),
  `ask_admin` (post a question into the Telegram chat and wait), or `abort`.

Every decision is streamed into the Telegram chat. Set `CC_SUPERVISOR=false` to
disable it and run plans straight through.

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

### Recommended hybrid: uncensored base on your GPU + frontier specialists

```bash
# On the cloud GPU box: serve the uncensored open-weight base (the brain)
ollama serve &
ollama pull huihui_ai/qwen3-abliterated:32b   # needs a GPU for full precision
# in .env:
#   MODEL_BASE_URL=http://your-gpu-host:11434/v1   # base/brain endpoint
#   MODEL_API_KEY=ollama
#   MODEL_PLANNER=huihui-ai/Qwen3-32B-abliterated  # uncensored base (no override)
#   # research/code/fix via one OpenRouter key:
#   MODEL_REASONER=google/gemini-2.5-pro      MODEL_REASONER_BASE_URL=https://openrouter.ai/api/v1  MODEL_REASONER_API_KEY=sk-or-...
#   MODEL_CODER=anthropic/claude-opus-4.1     MODEL_CODER_BASE_URL=https://openrouter.ai/api/v1     MODEL_CODER_API_KEY=sk-or-...
#   MODEL_FIXER=google/gemini-2.5-flash       MODEL_FIXER_BASE_URL=https://openrouter.ai/api/v1     MODEL_FIXER_API_KEY=sk-or-...
```

### Running fully local (no external key)

Leave every role on the single local endpoint (no per-role overrides) and point
all four `MODEL_*` ids at locally served models. On a CPU-only / small box, swap in
a tiny model (e.g. `ollama pull dolphin-phi`) just to exercise the pipeline — the
brain quality will be limited.

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
