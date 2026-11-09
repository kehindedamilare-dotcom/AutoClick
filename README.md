# AutoClick — an AI Computer-Use Agent

A modular, watchdogged desktop agent that observes your screen, asks a
vision-language model what to do next, validates the answer against a strict
Pydantic schema, and executes it with PyAutoGUI.

Built for unattended-but-supervised use: there is a human-in-the-loop gate, a
global kill-switch, a cursor fail-safe, stagnation detection, and a full audit
trail of every thought, click, and keystroke.

```
        ┌──────────────────────────── OODA loop ────────────────────────────┐
        │                                                                    │
        │  OBSERVE   mss capture (optional ROI) → downscale → PNG/base64     │
        │  ORIENT    goal + last 3 actions + screenshot → prompt             │
        │  DECIDE    VLM call (tenacity retries) → JSON → Pydantic Action    │
        │  ACT       map pixels → logical coords → clamp → PyAutoGUI         │
        │  EVALUATE  frame diff, failures, stagnation cap                    │
        │                                                                    │
        └────────────────────────────────────────────────────────────────────┘
```

## Project layout

| File            | Responsibility                                                              |
| --------------- | --------------------------------------------------------------------------- |
| `config.py`     | Typed settings, `.env` loading, validation, logging bootstrap               |
| `schema.py`     | `ActionType`, `AgentAction` Pydantic model, JSON parsing, shared exceptions |
| `vision_llm.py` | Screen capture, ROI/DPI geometry, trajectory memory, system prompt, clients |
| `executor.py`   | Coordinate mapping/clamping, Bezier mouse paths, humanized typing           |
| `agent.py`      | Loop orchestration, kill-switch, HITL, stagnation, transcripts              |
| `main.py`       | CLI, diagnostics, exit codes                                                |

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ is required.

Then:

```bash
cp .env.example .env      # Windows: copy .env.example .env
# edit .env and set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
```

> **Platform notes**
> - **macOS:** grant Accessibility + Screen Recording permissions to the
>   terminal/IDE running the agent. Global hotkeys may also need Input
>   Monitoring permission.
> - **Linux:** the `keyboard` library usually needs root for global hooks; the
>   agent logs a warning and keeps running with PyAutoGUI fail-safe only.
> - **Windows:** no extra permissions are normally required.

## Run

```bash
# Inspect monitors, scaling and ROI bounds first
python main.py --diagnose

# Dry-ish first run: confirm every action yourself
python main.py "Open Notepad and type: hello from AutoClick" --hitl --max-steps 25

# Crop the capture to a window to save tokens and improve focus
python main.py "Summarize the document on screen" --roi 100,100,1400,900

# Unattended run with a step budget
python main.py "Save the open document as report.txt" --max-steps 25
```

Exit codes: `0` done, `1` error, `2` escalated/stagnated/budget, `130` aborted.

## How the loop works

1. **Observe** — `mss` grabs the selected monitor (or ROI), producing a
   *physical-pixel* image. The image is downscaled to `AGENT_MAX_IMAGE_SIDE`.
2. **Orient** — the prompt contains the goal, the step number, the screenshot,
   and a JSON summary of the last three actions including whether the screen
   changed after each one.
3. **Decide** — `VisionLLM.decide()` calls the model with `response_format=
   json_object` (OpenAI), retries transient failures with exponential backoff
   (`tenacity`), and validates the result with Pydantic. If the model emits
   broken JSON, it gets up to `AGENT_SCHEMA_REPAIR_ATTEMPTS` corrective
   round-trips before the step fails.
4. **Act** — coordinates are translated from the model's *normalized 0–1000
   grid* into *screenshot pixels*, then to *logical desktop points*, clamped 5 px
   inside the display edges, and executed with randomized Bezier mouse paths and
   per-keystroke delays.
5. **Evaluate** — the next observation is diffed against the previous one. At
   startup the agent samples the idle screen and masks the pixels that animate
   by themselves (text carets, spinners, clocks), so a blinking cursor is not
   mistaken for progress and a single typed line still registers; an apparently
   unchanged screen is re-grabbed briefly in case the UI repainted late. A run of
   identical actions, or one action retried with no effect, triggers
   `StagnationError`; repeated executor failures trigger escalation.

## Coordinate pipeline (the part that usually breaks)

Modern displays break naive automation: `mss` returns **physical** pixels, while
PyAutoGUI addresses **logical** points. On Windows at 150% scaling a 3840 px
screen is 2560 logical points; on a Retina Mac it is 2 physical pixels per point.

The model reports points on a **normalized 0–1000 grid**, not in pixels: this is
how vision-language models (Gemini especially) are trained to express points, and
asking for pixels is what makes clicks land tens of pixels off target. The
mapping applied to every model coordinate `(nx, ny)` is:

```
image    = (nx / 1000 * sent_width, ny / 1000 * sent_height)
physical = capture_origin + image * (captured_size / sent_size)
logical  = physical / display_scale
final    = clamp(logical, display_bounds ± boundary_buffer)
```

Values outside `0..1000` are rejected by the schema and sent back for repair, so
a model that answers in pixels is corrected instead of clicking at random.

- `AGENT_DISPLAY_SCALE` unset → auto-detected as `primary_physical_width /
  pyautogui.size().width` (e.g. `1.5` or `2.0`).
- `AGENT_ROI` is in **physical** pixels, i.e. exactly the numbers printed by
  `--diagnose`. Cropping and downscaling are both compensated for.
- Mixed-DPI multi-monitor setups have no single global scale; set
  `AGENT_MONITOR_INDEX` to the monitor you care about and pin
  `AGENT_DISPLAY_SCALE` to that monitor's scaling factor.

## Choosing a model provider

`AGENT_PROVIDER` selects the vision backend; each provider reads its own key
(`AGENT_API_KEY` overrides any of them).

| Provider | `AGENT_PROVIDER` | Key variable | Default model |
| --- | --- | --- | --- |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| Google Gemini | `google` | `GOOGLE_API_KEY` | `gemini-2.5-flash` |

To use a Gemini key, get one from <https://aistudio.google.com/apikey> and set:

```bash
AGENT_PROVIDER=google
GOOGLE_API_KEY=your-key-here
# optional: AGENT_MODEL=gemini-2.5-pro
```

## Configuration reference

Every field below can be set by the environment variable in the right column
(or in `.env`); most are also exposed as CLI flags. See `config.py` for the
authoritative list, defaults, and validation rules.

| Env var | Meaning |
| --- | --- |
| `AGENT_PROVIDER` | `openai`, `anthropic`, or `google` (Gemini) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Credentials (or `AGENT_API_KEY`) |
| `AGENT_MODEL` | Vision-capable model id |
| `AGENT_BASE_URL` | OpenAI-compatible gateway URL |
| `AGENT_TEMPERATURE` | Sampling temperature (default `0.0`) |
| `AGENT_JSON_MODE` | Use provider JSON mode when available |
| `AGENT_HITL` | Require `y/n` confirmation for every action |
| `AGENT_MAX_STEPS` | Step budget per run (default 25); also `--max-steps N` |
| `AGENT_MAX_RUNTIME_SECONDS` | Wall-clock budget per run (default 900) |
| `AGENT_KILL_SWITCH_ENABLED` | Arm the global hotkey |
| `AGENT_KILL_SWITCH_HOTKEY` | Default `ctrl+alt+k` |
| `AGENT_FAILSAFE` | PyAutoGUI corner fail-safe |
| `AGENT_MAX_IDENTICAL_ACTIONS` | Stagnation threshold (default 3) |
| `AGENT_MAX_STATE_REPEATS` | Times an action may be repeated from the same screen before it is rejected (default 2) |
| `AGENT_MAX_FUTILE_ATTEMPTS` | Retries of one action that changes nothing (default 2) |
| `AGENT_MAX_CONSECUTIVE_FAILURES` | Escalate after N driver errors |
| `AGENT_MONITOR_INDEX` | `1` = primary, `2` = second display |
| `AGENT_ROI` | `left,top,width,height` in physical pixels |
| `AGENT_DISPLAY_SCALE` | Override DPI multiplier |
| `AGENT_MAX_IMAGE_SIDE` | Longest screenshot side in pixels |
| `AGENT_BOUNDARY_BUFFER` | Edge padding for clicks (default 5 px) |
| `AGENT_SCREEN_CHANGE_THRESHOLD` | Frame-diff sensitivity (0–255 mean) |
| `AGENT_SCREEN_CHANGE_FRACTION` | Share of changed pixels that counts as a change (default 0.0005) |
| `AGENT_SCREEN_CHANGE_PIXEL_DELTA` | Per-pixel delta counted as changed (default 8) |
| `AGENT_SCREEN_CHANGE_RECHECKS` | Re-grabs before calling an action a no-op (default 2) |
| `AGENT_SCREEN_CHANGE_RECHECK_DELAY` | Seconds between those re-grabs (default 0.5) |
| `AGENT_KEY_DELAY_RANGE` | Per-keystroke delay range in seconds |
| `AGENT_MOVE_DURATION_RANGE` | Mouse travel duration range |
| `AGENT_MOVE_JITTER_PX` | Maximum mouse path jitter |
| `AGENT_PRE_CLICK_DELAY_RANGE` | Pause before pressing the button |
| `AGENT_SETTLE_RANGE` | Pause after each action |
| `AGENT_WAIT_RANGE` | Duration of a `WAIT` action |
| `AGENT_MAX_STEPS` | Loop budget |
| `AGENT_MAX_RUNTIME_SECONDS` | Wall-clock budget |
| `AGENT_SCROLL_DEFAULT` / `AGENT_SCROLL_MAX` | Scroll clicks |
| `AGENT_SAVE_SCREENSHOTS` | Save frames + transcript under `runs/` |
| `AGENT_LOG_LEVEL` / `AGENT_LOG_TO_FILE` | Logging verbosity / file sink |

## Action space

The model must return one JSON object:

```json
{
  "thought_process": "The Save As dialog is open; the filename field is focused.",
  "action_type": "TYPE",
  "coordinates": null,
  "text_input": "report.txt"
}
```

| `action_type` | Requirements |
| --- | --- |
| `CLICK` / `DOUBLE_CLICK` | exactly one `[x, y]` on the 0–1000 grid |
| `TYPE` | non-empty `text_input`; optional single click target |
| `PRESS_KEY` | non-empty `key_input`: one key or a `+` chord (`win`, `enter`, `ctrl+s`); no coordinates |
| `SCROLL` | `scroll_amount` in `[-15, 15]`; optional hover point |
| `DRAG` | exactly two points `[[x1,y1],[x2,y2]]` |
| `WAIT` | no coordinates |
| `DONE` | no coordinates |
| `ESCALATE_TO_HUMAN` | no coordinates; optional reason in `text_input` |

Invalid shapes are rejected before any OS call, and the model is asked to repair
its output.

## Safety model

- **Global kill-switch** (`Ctrl+Alt+K`): a background `keyboard` hook calls
  `os._exit(130)` — it works even while the loop is blocked in a sleep, an API
  call, or a HITL prompt.
- **PyAutoGUI fail-safe**: moving the cursor into a screen corner raises
  `FailSafeException`, which stops the run with exit code 130.
- **HITL mode**: prints the model's reasoning and exact action, then waits for
  `y/n/q`.
- **Stagnation cap**: three identical actions in a row raise `StagnationError`,
  log a critical escalation banner, and stop the loop. The duplicate check uses
  exact coordinates/typed text, so a slightly different action resets it.
- **Visited-screen memory**: every screen is fingerprinted, so returning to one
  the agent has already seen injects a note listing what it tried there and what
  it then observed. This is what stops A/B/A/B loops — click a paid template,
  close it, click it again — that never repeat an action twice in a row and are
  therefore invisible to the stagnation cap. If the same action is taken from
  the same screen `AGENT_MAX_STATE_REPEATS` times, it is rejected, the model is
  asked again with the evidence attached, and a second refusal ends the run.
- **Failure cap**: three consecutive executor errors escalate instead of
  hammering the desktop.
- **Prompt-level guardrails**: the system prompt forbids typing credentials,
  confirming destructive actions, bypassing CAPTCHAs, or touching terminals.
  These are instructions, not enforcement — review unattended runs.
- **Boundary clamp**: every coordinate is kept `AGENT_BOUNDARY_BUFFER` pixels
  away from every edge, which avoids Hot Corners and off-screen monitors.

> The kill-switch and fail-safe stop *this process*. They cannot undo actions
> already performed. Run the agent in a VM or dedicated desktop session when
> experimenting with irreversible workflows.

## Logs and artifacts

Each run creates `runs/run-<UTC timestamp>/` containing:

- `step_001.png` … — exactly what the model saw;
- `agent.log` — timestamped reasoning, decisions, mapped coordinates, clicks,
  keystrokes (character-level only at `DEBUG` so incidental secrets are not
  written to INFO logs);
- `transcript.json` — goal, configuration (no secrets), status, and every action
  with its outcome.

Disable frame capture with `--no-screenshots` (or `AGENT_SAVE_SCREENSHOTS=false`).
Set `AGENT_LOG_TO_FILE=false` as well to skip the run directory entirely.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `429` with `insufficient_quota` / "no credits remaining" | The provider account is out of credits. Add credits in its billing dashboard, or switch to another key/provider (`AGENT_API_KEY`, `AGENT_PROVIDER`, `AGENT_BASE_URL`). This is not a rate limit and is not retried |
| `401` rejected API key | The key is missing, revoked, or from a different provider than `AGENT_PROVIDER` |
| `404` model not found | The configured model name is wrong or unavailable to this key; set `AGENT_MODEL` to a vision-capable model |
| Clicks land offset from the target | Run `--diagnose`; pin `AGENT_DISPLAY_SCALE` to the printed ratio |
| Clicks land on the wrong monitor | Set `AGENT_MONITOR_INDEX`; remember ROI is in physical pixels |
| `Monitor index N is unavailable` | `--diagnose` lists valid indices |
| Kill-switch never fires | macOS/Linux permissions; the warning in the log says so. Fail-safe still works |
| Model keeps emitting invalid JSON | Set `AGENT_JSON_MODE=false` for gateways without JSON mode, or use a stronger model |
| Screenshot is black/locked | The session is locked or on a secure desktop; unlock before running |
| Agent repeats one click | That is the stagnation guard doing its job — read the transcript and refine the goal |
