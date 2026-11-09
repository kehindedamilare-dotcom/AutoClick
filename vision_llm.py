"""Perception and decision layer of the OODA loop.

Responsibilities:
    * Capture the screen with ``mss`` (multi-monitor aware, optional ROI crop).
    * Downscale frames for token efficiency while preserving a coordinate mapping.
    * Keep a short "trajectory memory" of recent actions.
    * Prompt a vision-language model and validate its JSON action with Pydantic,
      retrying transient transport failures with exponential backoff.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import mss
import pyautogui
from PIL import Image
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import openai
from openai import OpenAI

from config import DisplaySettings, LLMSettings, Roi
from schema import (
    AgentAction,
    ActionType,
    ActionValidationError,
    CaptureError,
    ConfigurationError,
    build_repair_message,
    parse_action,
)

try:  # Anthropic is an optional provider dependency.
    import anthropic
    from anthropic import Anthropic
except Exception:  # pragma: no cover - exercised only when SDK is absent
    anthropic = None  # type: ignore[assignment]
    Anthropic = None  # type: ignore[assignment]

try:  # Google Gemini is an optional provider dependency.
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - exercised only when the SDK is absent
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


def _decode_inline_image(media_type: str, data: str) -> bytes:
    """Decode the base64 payload of a ``data:`` URI into raw image bytes."""
    if not media_type.startswith("image/") or not data:
        return b""
    try:
        return base64.b64decode(data, validate=False)
    except Exception:  # pragma: no cover - malformed data URI
        LOGGER.debug("Could not decode inline %s image", media_type, exc_info=True)
        return b""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are "AutoClick", an autonomous computer-use agent driving a real desktop
through a mouse and keyboard. Each turn you receive exactly ONE screenshot, the
user's ultimate goal, and a short memory of your recent actions. You must choose
exactly ONE next action.

## 1. Observation rules
- The screenshot is ground truth. Do not rely on assumptions about what an app
  "usually" looks like; read the pixels in front of you.
- The image may be cropped to a window and may be downscaled. It is still the
  full visible universe for this step. If the target is not visible, scroll,
  open a menu, or navigate there instead of guessing.
- If the screen looks mid-animation, blurred, or still loading, choose WAIT.
- Never claim an action succeeded. Only state DONE when the screenshot *shows*
  the goal is satisfied.

## 2. Coordinate contract
- Coordinates are on a NORMALIZED 0-1000 grid, NOT pixels. Both values are
  integers from 0 to 1000:
    x = 0 is the left edge of the screenshot,   x = 1000 is the right edge.
    y = 0 is the top edge of the screenshot,    y = 1000 is the bottom edge.
  Convert what you see: x = round(pixel_x / image_width * 1000), and likewise
  for y against the image height. A control in the middle of the screen is
  about [500, 500]; one just below a full-width header is about [200, 150].
- Aim at the visual centre of the target (button label, menu row, text field
  interior). Never click text labels when the interactive control is elsewhere.
- Never return a value above 1000 or below 0. If unsure, WAIT instead.
- For DRAG, return exactly two points: [[x_start, y_start], [x_end, y_end]].

## 3. Response contract
Return exactly ONE JSON object, no markdown fences and no prose before or after:

{
  "thought_process": "One or two sentences: what is visible and why this action.",
  "action_type": "CLICK | DOUBLE_CLICK | TYPE | PRESS_KEY | SCROLL | DRAG | WAIT | DONE | ESCALATE_TO_HUMAN",
  "coordinates": [[x, y]] | [[x1, y1], [x2, y2]] | null,   // 0-1000 grid
  "text_input": "string or null",
  "key_input": "string or null",
  "scroll_amount": 3
}

Field rules:
- thought_process: required, concrete, grounded in visible evidence, <= 40 words.
- CLICK / DOUBLE_CLICK: exactly one coordinate.
- TYPE: text_input required (exact text to enter). The text goes to whichever
  control currently has focus, so omit the coordinate when the target was just
  opened or launched; it already has focus. Only add a coordinate when you must
  focus a specific field you can see inside the already-active window. Never
  click-into a window belonging to another application.
- PRESS_KEY: key_input required, one key ("win", "enter", "esc", "tab", "f5") or
  a chord joined by '+' ("ctrl+s", "alt+tab", "ctrl+shift+t"). No coordinates.
  Prefer this over clicking when the control you need is not visible.
- SCROLL: scroll_amount from -15 to 15; positive scrolls content up (wheel away
  from the user), negative scrolls down. Optional single coordinate to hover.
- DRAG: exactly two coordinates, start then end.
- WAIT: no coordinates. Use for loading or animations.
- DONE: no coordinates. Use only when the goal is visibly complete.
- ESCALATE_TO_HUMAN: no coordinates. Optional text_input explaining what the
  human must do. Use when blocked.
- Never combine two actions in one step; a key chord belongs in a single
  PRESS_KEY, not a sequence such as "win,enter".

## 4. Decision policy
1. Re-read the goal and the recent trajectory. If the last action failed or the
   screen did not change, change strategy instead of repeating it verbatim.
   Never repeat a click that produced no visible change; a repeated identical
   action is rejected as stagnation. Switch to a keyboard path instead: the Run
   dialog, a shortcut you know, or tab/enter navigation. If a click opens a
   context menu instead of activating its target, press "esc" and reach the
   target by keyboard; do not keep clicking the same place.
2. Prefer the shortest reliable path: keyboard shortcuts or direct typing beat
   hunting through menus. To search or open a site in a browser, never click
   into the page: PRESS_KEY "ctrl+l" focuses the address bar, then TYPE the
   query or URL, then PRESS_KEY "enter". "ctrl+t" opens a tab and "ctrl+w"
   closes one.
3. To launch an application, prefer the Run dialog: PRESS_KEY "win+r", then
   TYPE the executable name ("notepad", "calc", "mspaint", "cmd"), then
   PRESS_KEY "enter". This works even when the taskbar is hidden and no icon is
   visible. Click a Start-menu tile only if the Run dialog is unavailable.
4. A freshly launched application owns the keyboard focus: type into it with no
   click coordinate. Clicking a window does not reliably focus it on every
   system, and text then lands in the wrong app.
5. A window can be visible without owning the keyboard. Before typing into an
   existing window, make it active with PRESS_KEY "alt+tab" (one tap per window
   switch, check the screenshot after each) rather than clicking its title bar or
   content: clicking a background window does not reliably focus it and your
   text then lands in the wrong application. Keyboard shortcuts only reach the
   active window.
6. After clicking a text field, verify focus (caret, highlight) before typing.
   When you type into a field, assume the previous step already entered the
   text; do not retype the same string unless you can see it is missing.
7. If a dialog, popup, cookie banner, or update prompt blocks the task, dismiss
   it with the least destructive available control first.
8. If an action appears to have had no effect twice, try one genuinely
   different approach; after that, escalate.
9. Keep each step small and observable. Prefer a click plus a later TYPE over a
   click that also submits a form.

## 5. Hard safety rules
- Never type passwords, payment card numbers, API keys, or 2FA codes. If a
  credential is required, ESCALATE_TO_HUMAN.
- Never confirm irreversible or destructive actions (delete, format, purchase,
  send money, publish, accept legal terms, close unsaved work) on your own.
  ESCALATE_TO_HUMAN when such a confirmation appears.
- Never attempt to bypass CAPTCHAs, OS security prompts, or antivirus warnings.
- Never use developer consoles, terminal commands, or scripts that could exfiltrate
  data or modify the system beyond what the goal requires.
- If the screen shows something unexpected (wrong app, wrong account, sensitive
  data), stop and ESCALATE_TO_HUMAN instead of improvising.

## 6. Failure handling
- Invalid JSON is rejected. If asked to repair, return the full corrected object.
- If the task is impossible from this screen, ESCALATE_TO_HUMAN with a precise
  reason rather than clicking randomly.

Think briefly inside thought_process, then emit the single JSON object.
"""


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenGeometry:
    """Physical-to-logical mapping metadata for the current capture setup."""

    monitor_index: int
    monitor_left: int
    monitor_top: int
    monitor_width: int
    monitor_height: int
    virtual_left: int
    virtual_top: int
    virtual_width: int
    virtual_height: int
    scale: float
    roi: Optional[Roi]

    @property
    def capture_region(self) -> Roi:
        """``(left, top, width, height)`` in physical pixels passed to MSS."""
        if self.roi is not None:
            return self.roi
        return (self.monitor_left, self.monitor_top, self.monitor_width, self.monitor_height)

    @property
    def logical_bounds(self) -> tuple[int, int, int, int]:
        """Virtual desktop bounds in *logical* (PyAutoGUI) coordinates."""
        scale = self.scale if self.scale > 0 else 1.0
        return (
            round(self.virtual_left / scale),
            round(self.virtual_top / scale),
            max(1, round(self.virtual_width / scale)),
            max(1, round(self.virtual_height / scale)),
        )


@dataclass(frozen=True)
class ScreenFrame:
    """One captured screenshot plus everything needed to map coordinates back."""

    image: Image.Image
    captured_width: int
    captured_height: int
    origin_x: int
    origin_y: int
    scale: float
    captured_at: float = field(default_factory=time.time)

    @property
    def sent_width(self) -> int:
        return self.image.width

    @property
    def sent_height(self) -> int:
        return self.image.height

    def to_png_bytes(self) -> bytes:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def to_data_url(self) -> str:
        encoded = base64.b64encode(self.to_png_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def prompt_summary(self) -> str:
        return (
            f"{self.sent_width}x{self.sent_height} px screenshot "
            f"(downscaled from a {self.captured_width}x{self.captured_height} px capture)"
        )


class ScreenCapture:
    """Thread-confined screen grabber built on MSS with ROI and downscaling."""

    def __init__(self, settings: DisplaySettings) -> None:
        self._settings = settings
        self._sct = mss.mss()
        try:
            self._monitors: list[dict[str, int]] = [dict(m) for m in self._sct.monitors]
        except Exception as exc:  # pragma: no cover - driver specific
            self._sct.close()
            raise CaptureError(f"Could not enumerate monitors: {exc}") from exc

        if not 1 <= settings.monitor_index < len(self._monitors):
            self._sct.close()
            raise ConfigurationError(
                f"Monitor index {settings.monitor_index} is unavailable; "
                f"found {len(self._monitors) - 1} display(s)."
            )

        self._monitor = self._monitors[settings.monitor_index]
        self._scale = settings.display_scale or self._detect_scale()
        self._region = self._resolve_region(settings.roi)
        self._geometry = ScreenGeometry(
            monitor_index=settings.monitor_index,
            monitor_left=self._monitor["left"],
            monitor_top=self._monitor["top"],
            monitor_width=self._monitor["width"],
            monitor_height=self._monitor["height"],
            virtual_left=self._monitors[0]["left"],
            virtual_top=self._monitors[0]["top"],
            virtual_width=self._monitors[0]["width"],
            virtual_height=self._monitors[0]["height"],
            scale=self._scale,
            roi=settings.roi,
        )
        LOGGER.info(
            "Capture ready: monitor=%d region=%s scale=%.3f max_side=%d",
            settings.monitor_index,
            self._region,
            self._scale,
            settings.max_image_side,
        )

    # -- properties ---------------------------------------------------------

    @property
    def geometry(self) -> ScreenGeometry:
        return self._geometry

    @property
    def monitors(self) -> Sequence[Mapping[str, int]]:
        return tuple(self._monitors)

    # -- setup helpers ------------------------------------------------------

    def _detect_scale(self) -> float:
        """Physical pixels per logical point, derived from the primary display."""
        try:
            logical_width, _ = pyautogui.size()
        except Exception as exc:  # pragma: no cover - driver specific
            LOGGER.warning("Could not read logical screen size (%s); assuming scale=1.", exc)
            return 1.0
        primary = self._monitors[1] if len(self._monitors) > 1 else self._monitors[0]
        physical_width = int(primary["width"])
        if logical_width <= 0 or physical_width <= 0:
            return 1.0
        ratio = physical_width / logical_width
        if ratio <= 0:
            return 1.0
        LOGGER.debug(
            "Auto-detected display scale %.4f (physical %d / logical %d)",
            ratio,
            physical_width,
            logical_width,
        )
        return round(ratio, 4)

    def _resolve_region(self, roi: Optional[Roi]) -> Roi:
        virtual = self._monitors[0]
        if roi is None:
            return (
                int(self._monitor["left"]),
                int(self._monitor["top"]),
                int(self._monitor["width"]),
                int(self._monitor["height"]),
            )
        left, top, width, height = roi
        if width <= 0 or height <= 0:
            raise ConfigurationError("ROI width and height must be positive.")
        right, bottom = left + width, top + height
        v_right = virtual["left"] + virtual["width"]
        v_bottom = virtual["top"] + virtual["height"]
        if left < virtual["left"] or top < virtual["top"] or right > v_right or bottom > v_bottom:
            raise ConfigurationError(
                f"ROI {roi} lies outside the virtual desktop "
                f"({virtual['left']}, {virtual['top']}, "
                f"{virtual['width']}, {virtual['height']})."
            )
        return (left, top, width, height)

    # -- capture ------------------------------------------------------------

    def grab(self) -> ScreenFrame:
        """Capture, crop, and downscale one frame."""
        try:
            shot = self._sct.grab(self._region)  # type: ignore[arg-type]
        except Exception as exc:
            raise CaptureError(f"Screen capture failed for region {self._region}: {exc}") from exc

        try:
            image = Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception as exc:
            raise CaptureError(f"Could not convert screen buffer to an image: {exc}") from exc

        captured_width, captured_height = image.size
        max_side = self._settings.max_image_side
        longest = max(captured_width, captured_height)
        if longest > max_side:
            ratio = max_side / float(longest)
            target = (max(1, round(captured_width * ratio)), max(1, round(captured_height * ratio)))
            image = image.resize(target, Image.Resampling.LANCZOS)
            LOGGER.debug(
                "Downscaled screenshot %dx%d -> %dx%d",
                captured_width,
                captured_height,
                *target,
            )

        return ScreenFrame(
            image=image,
            captured_width=captured_width,
            captured_height=captured_height,
            origin_x=self._region[0],
            origin_y=self._region[1],
            scale=self._scale,
        )

    def save_frame(self, frame: ScreenFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frame.to_png_bytes())
        return path

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:  # pragma: no cover - best effort
            LOGGER.debug("Ignoring error while closing MSS", exc_info=True)


def describe_environment(capture: ScreenCapture) -> str:
    """Human-readable diagnostics used by ``python main.py --diagnose``."""
    geometry = capture.geometry
    try:
        logical_size = pyautogui.size()
    except Exception as exc:  # pragma: no cover
        logical_size = None
        LOGGER.warning("pyautogui.size() failed: %s", exc)

    lines = [
        "Display diagnostics",
        "-------------------",
        f"Logical screen (PyAutoGUI): {logical_size}",
        f"Physical / logical scale : {geometry.scale:.4f}",
        f"Virtual desktop (physical): left={geometry.virtual_left} top={geometry.virtual_top} "
        f"width={geometry.virtual_width} height={geometry.virtual_height}",
        f"Logical desktop bounds   : {geometry.logical_bounds}",
        f"Selected monitor         : #{geometry.monitor_index} "
        f"left={geometry.monitor_left} top={geometry.monitor_top} "
        f"{geometry.monitor_width}x{geometry.monitor_height}",
        f"ROI (physical pixels)    : {geometry.roi or 'none (full monitor)'}",
        f"Capture region           : {geometry.capture_region}",
        "",
        "MSS monitors:",
    ]
    for index, monitor in enumerate(capture.monitors):
        label = "virtual" if index == 0 else f"display {index}"
        lines.append(
            f"  [{index}] {label:<9} left={monitor['left']:>6} top={monitor['top']:>6} "
            f"{monitor['width']}x{monitor['height']}"
        )
    lines.append("")
    lines.append(
        "Set AGENT_ROI in physical pixels (left,top,width,height) to crop the capture."
    )
    lines.append(
        "Set AGENT_DISPLAY_SCALE to override auto-detection on mixed-DPI multi-monitor rigs."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trajectory memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """One executed step stored for the next VLM prompt."""

    step: int
    action: AgentAction
    outcome: str
    screen_changed: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        text = self.action.text_input or ""
        if len(text) > 200:
            text = text[:197] + "..."
        return {
            "step": self.step,
            "action": self.action.action_type.value,
            "coordinates": self.action.coordinates,
            "scroll_amount": self.action.scroll_amount,
            "text_input": text or None,
            "outcome": self.outcome,
            "screen_changed": self.screen_changed,
        }


class TrajectoryMemory:
    """Bounded history of recent actions used for orientation and stagnation."""

    def __init__(self, maxlen: int = 8) -> None:
        if maxlen < 1:
            raise ValueError("TrajectoryMemory maxlen must be >= 1")
        self._entries: deque[MemoryEntry] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def record(self, step: int, action: AgentAction, outcome: str) -> MemoryEntry:
        entry = MemoryEntry(step=step, action=action, outcome=outcome)
        self._entries.append(entry)
        return entry

    def update_last_screen_change(self, changed: bool) -> None:
        if self._entries:
            self._entries[-1].screen_changed = changed

    def reset(self) -> None:
        self._entries.clear()

    def prompt_block(self, last: int = 3) -> str:
        """Compact JSON block describing the most recent actions."""
        recent = list(self._entries)[-last:]
        if not recent:
            return "No actions have been taken yet."
        return json.dumps([entry.to_dict() for entry in recent], ensure_ascii=False, indent=2)

    def trailing_identical(self) -> int:
        """Length of the run of identical action signatures at the tail."""
        if not self._entries:
            return 0
        signatures = [entry.action.signature() for entry in self._entries]
        last = signatures[-1]
        count = 1
        for signature in reversed(signatures[:-1]):
            if signature != last:
                break
            count += 1
        return count

    def is_stagnant(self, limit: int) -> bool:
        """True when the last ``limit`` actions are exactly identical."""
        return self.trailing_identical() >= max(2, limit)


# ---------------------------------------------------------------------------
# Screen history (visited-state memory)
# ---------------------------------------------------------------------------

#: Two frames whose average pixel distance is below this are "the same screen".
STATE_TOLERANCE: float = 3.0
#: Grid units (of the 0-1000 coordinate space) within which two clicks count as
#: "the same target", so a wobbling click still matches the earlier attempt.
REPEAT_RADIUS_UNITS: float = 60.0


def _clip(text: str, limit: int) -> str:
    """Shorten a sentence without cutting a word in half."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if boundary > limit // 2:
        return cut[: boundary + 1]
    return cut.rsplit(" ", 1)[0] + "..."


@dataclass
class AttemptRecord:
    """An action taken from one screen state, plus what the model saw after it."""

    step: int
    action: AgentAction
    # Thoughts recorded while looking at the result. Two are kept because an
    # action often produces a loading screen first and only the second look
    # reveals what actually came up (a paid template, an error, an empty list).
    thoughts_after: list[str] = field(default_factory=list)
    thoughts_remaining: int = 2

    @property
    def consequence(self) -> str:
        return " | ".join(self.thoughts_after)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "step": self.step,
            "action": self.action.action_type.value,
            "coordinates": self.action.coordinates,
            "text_input": (self.action.text_input or None),
            "key_input": self.action.key_input,
        }
        if self.thoughts_after:
            data["next_thought"] = self.consequence[:300]
        return data


@dataclass
class StateVisit:
    """One distinct screen state and everything tried from it."""

    fingerprint: bytes
    steps: list[int] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def revisits(self) -> bool:
        return len(self.steps) > 1


class ScreenHistory:
    """Remembers screens the agent has already seen and what it tried there.

    Trajectory memory only knows recent *actions*, so once a modal closes and the
    screen returns to an identical grid the agent has no way to know it already
    learned something there. This keeps a small perceptual fingerprint of every
    visited screen and the attempts made from it, so a revisit can be reported as
    "you tried this here already, and it led to a paid template" instead of being
    re-derived from pixels.
    """

    def __init__(
        self,
        max_states: int = 40,
        tolerance: float = STATE_TOLERANCE,
        repeat_radius: float = REPEAT_RADIUS_UNITS,
    ) -> None:
        self._max_states = max(1, max_states)
        self._tolerance = tolerance
        self._repeat_radius = repeat_radius
        self._visits: list[StateVisit] = []
        # Attempts made in the last couple of steps. A thought is attached to all
        # of them, because the screen right after a click is often only a loading
        # state and the meaningful consequence arrives one step later.
        self._recent: deque[AttemptRecord] = deque(maxlen=3)

    # -- fingerprints -------------------------------------------------------

    @staticmethod
    def fingerprint(thumbnail: Image.Image) -> bytes:
        """Coarse grayscale signature; small enough to compare point by point."""
        return thumbnail.convert("L").resize((32, 18), Image.Resampling.BILINEAR).tobytes()

    def _distance(self, left: bytes, right: bytes) -> float:
        if len(left) != len(right):
            return float("inf")
        return sum(abs(a - b) for a, b in zip(left, right)) / len(left)

    # -- observation --------------------------------------------------------

    def observe(self, step: int, thumbnail: Image.Image) -> StateVisit:
        """Return the visit for this screen, creating it the first time."""
        fingerprint = self.fingerprint(thumbnail)
        best: Optional[StateVisit] = None
        best_distance = float("inf")
        for visit in self._visits:
            distance = self._distance(visit.fingerprint, fingerprint)
            if distance < best_distance:
                best, best_distance = visit, distance
        if best is not None and best_distance <= self._tolerance:
            best.steps.append(step)
            return best

        visit = StateVisit(fingerprint=fingerprint, steps=[step])
        self._visits.append(visit)
        if len(self._visits) > self._max_states:
            # Forget the least recently seen screen, keeping the current one.
            oldest = min(
                (v for v in self._visits if v is not visit),
                key=lambda v: v.steps[-1] if v.steps else 0,
                default=None,
            )
            if oldest is not None:
                self._visits.remove(oldest)
        return visit

    # -- attempts -----------------------------------------------------------

    def note_thought(self, thought: str) -> None:
        """Attach the current reasoning to the attempts that produced this screen.

        Called after the model decides on the *next* step: its thought at that
        point describes what the previous action produced, which is exactly the
        consequence a revisit needs. Two thoughts per attempt are kept, because
        the screen right after a click is often just a loading state.
        """
        if not thought:
            return
        for record in self._recent:
            if record.thoughts_remaining > 0:
                record.thoughts_after.append(thought.strip())
                record.thoughts_remaining -= 1

    @staticmethod
    def _target_distance(left: AgentAction, right: AgentAction) -> float:
        """Distance in grid units between two actions' targets (0 when equal)."""
        if not left.coordinates or not right.coordinates:
            same_input = (
                left.key_input == right.key_input
                and (left.text_input or "") == (right.text_input or "")
            )
            return 0.0 if same_input else float("inf")
        (lx, ly), (rx, ry) = left.coordinates[0], right.coordinates[0]
        return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5

    def prior_attempts(self, visit: StateVisit, action: AgentAction) -> list[AttemptRecord]:
        """Earlier attempts from this screen that targeted the same place the same way."""
        return [
            record
            for record in visit.attempts
            if record.action.action_type is action.action_type
            and self._target_distance(record.action, action) <= self._repeat_radius
        ]

    def record_action(self, visit: StateVisit, step: int, action: AgentAction) -> int:
        """Store an attempt and return how many similar ones preceded it."""
        prior = len(self.prior_attempts(visit, action))
        record = AttemptRecord(step=step, action=action)
        visit.attempts.append(record)
        self._recent.append(record)
        return prior

    # -- prompting ----------------------------------------------------------

    def prompt_block(self, visit: StateVisit) -> str:
        """Describe earlier attempts from a revisited screen, or an empty string."""
        if not visit.revisits:
            return ""
        # Waiting is not a target, and an attempt with no recorded consequence
        # teaches nothing ("you clicked here" alone is not actionable).
        attempts = [
            record
            for record in visit.attempts
            if record.action.action_type is not ActionType.WAIT and record.thoughts_after
        ]
        if not attempts:
            return ""
        lines = [
            "## SCREEN ALREADY VISITED",
            "You are on a screen you have been on before (steps %s)."
            % ", ".join(str(step) for step in visit.steps[:-1][-4:]),
            "From this exact screen you already tried:",
        ]
        for record in attempts[-4:]:
            target = ""
            if record.action.coordinates:
                x, y = record.action.coordinates[0]
                target = f" at ({x}, {y})"
            elif record.action.key_input:
                target = f" key={record.action.key_input!r}"
            elif record.action.text_input:
                target = f" text={record.action.text_input[:40]!r}"
            line = f"- step {record.step}: {record.action.action_type.value}{target}"
            if record.thoughts_after:
                line += f" -> you then observed: \"{_clip(record.consequence, 280)}\""
            lines.append(line)
        lines.append(
            "Repeating any of these will repeat its result. Choose a different "
            "target, or a different kind of action, instead of the same one again."
        )
        return "\n".join(lines)

    def describe_visit(self, visit: StateVisit) -> str:
        """Short log line for ORIENT."""
        if not visit.revisits:
            return "new"
        return "revisit of steps " + ", ".join(str(step) for step in visit.steps[:-1][-4:])


# ---------------------------------------------------------------------------
# Vision-language model client
# ---------------------------------------------------------------------------


class VisionLLM:
    """OpenAI/Anthropic/Gemini vision client with tenacity retries and schema repair."""

    # Provider error text fragments that mean "this account cannot pay for
    # requests". Retrying these is pointless and only delays a clear message.
    _QUOTA_MARKERS: tuple[str, ...] = (
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
        "no credits remaining",
        "exceeded your current quota",
        "quota exceeded",
    )

    _KEY_ENV_VARS: Mapping[str, str] = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    def __init__(self, settings: LLMSettings, memory: Optional[TrajectoryMemory] = None) -> None:
        self._settings = settings
        self._memory = memory if memory is not None else TrajectoryMemory()
        if not settings.api_key:
            key_var = self._KEY_ENV_VARS.get(settings.provider, "AGENT_API_KEY")
            raise ConfigurationError(
                f"No API key configured. Add {key_var} (or AGENT_API_KEY) to your environment."
            )
        self._client = self._build_client()
        self._request: Callable[[Sequence[Mapping[str, Any]]], str] = self._build_retryer()(
            self._request_once
        )
        LOGGER.info(
            "VisionLLM ready: provider=%s model=%s base_url=%s",
            settings.provider,
            settings.model,
            settings.base_url or "default",
        )

    # -- construction -------------------------------------------------------

    def _build_client(self) -> Any:
        provider = self._settings.provider
        if provider == "openai":
            return OpenAI(
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
                timeout=self._settings.request_timeout,
                max_retries=0,  # retries are owned by tenacity
            )
        if provider == "anthropic":
            if Anthropic is None:
                raise ConfigurationError(
                    "AGENT_PROVIDER=anthropic but the 'anthropic' package is not installed."
                )
            return Anthropic(
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
                timeout=self._settings.request_timeout,
                max_retries=0,
            )
        if provider == "google":
            if genai is None:
                raise ConfigurationError(
                    "AGENT_PROVIDER=google but the 'google-genai' package is not installed."
                )
            return genai.Client(api_key=self._settings.api_key)
        raise ConfigurationError(f"Unsupported provider {provider!r}.")  # pragma: no cover

    def _transient_errors(self) -> tuple[type[BaseException], ...]:
        errors: list[type[BaseException]] = [
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ]
        if anthropic is not None:
            errors.extend(
                [
                    anthropic.RateLimitError,
                    anthropic.APITimeoutError,
                    anthropic.APIConnectionError,
                    anthropic.InternalServerError,
                ]
            )
        if self._settings.provider == "google" and genai is not None:
            from google.genai import errors as genai_errors

            errors.extend([genai_errors.APIError])
        return tuple(errors)

    def _should_retry(self, exc: BaseException) -> bool:
        """Retry only genuinely transient failures, never billing/quota errors."""
        if not isinstance(exc, self._transient_errors()):
            return False
        return self._explain_provider_error(exc) is None

    def _explain_provider_error(self, exc: BaseException) -> Optional[ConfigurationError]:
        """Translate permanent provider failures into actionable configuration errors."""
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        provider = self._settings.provider
        key_var = self._KEY_ENV_VARS.get(provider, "AGENT_API_KEY")

        if any(marker in text for marker in self._QUOTA_MARKERS):
            return ConfigurationError(
                f"The {provider} account for {key_var} has no remaining quota/credits "
                f"(HTTP {status or 429}). Add credits in the {provider} billing dashboard, "
                "or point the agent at another key/provider via AGENT_API_KEY, "
                "AGENT_PROVIDER, and AGENT_BASE_URL, then rerun."
            )

        if provider == "google":
            code = getattr(exc, "code", None) or status
            # Rate limits and 5xx responses are worth retrying; everything else
            # (bad key, missing model, malformed request) is not.
            if code in (429, 500, 502, 503, 504) or isinstance(
                exc, (TimeoutError, ConnectionError)
            ):
                return None
            if code in (401, 403) or "api key not valid" in text or "api_key_invalid" in text:
                return ConfigurationError(
                    "Google rejected the Gemini API key. Verify GOOGLE_API_KEY at "
                    "https://aistudio.google.com/apikey, then rerun."
                )
            if code == 404 or ("not found" in text and "model" in text):
                return ConfigurationError(
                    f"Gemini does not recognize model {self._settings.model!r}. "
                    "Pick a vision-capable model with --model or AGENT_MODEL "
                    "(for example gemini-2.5-flash)."
                )
            return ConfigurationError(
                f"Gemini request failed permanently: {exc}. "
                "Check GOOGLE_API_KEY, AGENT_MODEL, and the key's quota."
            )

        if status == 401:
            return ConfigurationError(
                f"{provider} rejected the API key (HTTP 401). Check {key_var} (or "
                "AGENT_API_KEY) in your .env file, then rerun."
            )
        if status == 403:
            return ConfigurationError(
                f"{provider} denied access (HTTP 403). The key may lack access to "
                f"model {self._settings.model!r} or to this endpoint."
            )
        if status == 404 and "model" in text:
            return ConfigurationError(
                f"{provider} does not recognize model {self._settings.model!r}. "
                "Pick a vision-capable model with --model or AGENT_MODEL."
            )
        return None

    def _build_retryer(self) -> Callable[[Callable[..., str]], Callable[..., str]]:
        """Create a tenacity decorator bound to the configured retry budget."""
        return retry(
            retry=retry_if_exception(self._should_retry),
            stop=stop_after_attempt(self._settings.api_max_attempts),
            wait=wait_random_exponential(
                multiplier=self._settings.retry_initial_wait,
                max=self._settings.retry_max_wait,
            ),
            before_sleep=before_sleep_log(LOGGER, logging.WARNING),
            reraise=True,
        )

    # -- public API ---------------------------------------------------------

    def decide(
        self, frame: ScreenFrame, goal: str, step: int, state_note: str = ""
    ) -> AgentAction:
        """Ask the model for one validated action, repairing bad JSON if needed."""
        messages: list[dict[str, Any]] = self._initial_messages(frame, goal, step, state_note)
        attempts = self._settings.schema_repair_attempts + 1
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            raw = self._request(messages)
            LOGGER.debug("Raw VLM response (schema attempt %d/%d): %s", attempt, attempts, raw)
            try:
                action = parse_action(raw)
            except ActionValidationError as exc:
                last_error = exc
                LOGGER.warning(
                    "VLM output failed validation (attempt %d/%d): %s", attempt, attempts, exc
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": build_repair_message(exc, raw)},
                ]
                continue
            LOGGER.info("DECIDE step=%d action=%s", step, action.describe())
            return action

        raise ActionValidationError(
            f"VLM did not return a valid action after {attempts} attempt(s): {last_error}"
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - best effort
                LOGGER.debug("Error while closing LLM client", exc_info=True)

    # -- prompts ------------------------------------------------------------

    def _initial_messages(
        self, frame: ScreenFrame, goal: str, step: int, state_note: str = ""
    ) -> list[dict[str, Any]]:
        user_text = self._build_user_text(frame, goal, step, state_note)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": frame.to_data_url(), "detail": "high"},
                    },
                ],
            },
        ]

    def _build_user_text(
        self, frame: ScreenFrame, goal: str, step: int, state_note: str = ""
    ) -> str:
        note = f"{state_note.strip()}\n\n" if state_note.strip() else ""
        return (
            f"GOAL:\n{goal.strip()}\n\n"
            f"STEP: {step}\n"
            f"SCREENSHOT: {frame.prompt_summary()}.\n"
            "All coordinates you return must be in the 0-1000 normalized grid "
            "described above (x right, y down), not pixels.\n\n"
            f"RECENT TRAJECTORY (oldest first, at most 3 entries):\n{self._memory.prompt_block()}\n\n"
            f"{note}"
            "Choose the single next action. Answer with one JSON object only."
        )

    # -- transport ----------------------------------------------------------

    def _request_once(self, messages: Sequence[Mapping[str, Any]]) -> str:
        try:
            if self._settings.provider == "openai":
                return self._openai_chat(messages)
            if self._settings.provider == "anthropic":
                return self._anthropic_chat(messages)
            if self._settings.provider == "google":
                return self._google_chat(messages)
            raise ConfigurationError(f"Unsupported provider {self._settings.provider!r}.")
        except ConfigurationError:
            raise
        except Exception as exc:
            friendly = self._explain_provider_error(exc)
            if friendly is not None:
                LOGGER.error("Provider request failed permanently: %s", friendly)
                raise friendly from exc
            raise

    def _openai_chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": list(messages),
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
            "timeout": self._settings.request_timeout,
        }
        if self._settings.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        if not completion.choices:
            raise ActionValidationError("provider returned no completion choices")
        content = completion.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if not content or not content.strip():
            raise ActionValidationError("provider returned an empty completion")
        return content

    def _anthropic_chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        system_prompt, converted = self._to_anthropic_messages(messages)
        response = self._client.messages.create(
            model=self._settings.model,
            system=system_prompt,
            messages=converted,
            max_tokens=self._settings.max_output_tokens,
            temperature=min(max(self._settings.temperature, 0.0), 1.0),
            timeout=self._settings.request_timeout,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise ActionValidationError("provider returned an empty completion")
        return text

    def _google_chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Call Gemini with the same message shape used for the other providers."""
        if genai is None or genai_types is None:  # pragma: no cover - build_client guards
            raise ConfigurationError(
                "AGENT_PROVIDER=google but the 'google-genai' package is not installed."
            )

        system_parts: list[str] = []
        contents: list[Any] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                continue

            parts: list[Any] = []
            if isinstance(content, str):
                parts.append(genai_types.Part.from_text(text=content))
            elif isinstance(content, Sequence):
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    if block.get("type") == "text":
                        parts.append(
                            genai_types.Part.from_text(text=str(block.get("text", "")))
                        )
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, Mapping) else ""
                        if not url.startswith("data:"):
                            continue
                        header, _, data = url.partition(",")
                        media_type = header[5:].split(";")[0] or "image/png"
                        image_bytes = _decode_inline_image(media_type, data)
                        if image_bytes:
                            parts.append(
                                genai_types.Part.from_bytes(
                                    data=image_bytes, mime_type=media_type
                                )
                            )

            if not parts:
                parts.append(genai_types.Part.from_text(text=" "))
            contents.append(
                genai_types.Content(
                    role="model" if role == "assistant" else "user", parts=parts
                )
            )

        if not contents:  # pragma: no cover - defensive
            contents = [
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=" ")])
            ]

        config: dict[str, Any] = {
            "temperature": self._settings.temperature,
            "max_output_tokens": self._settings.max_output_tokens,
        }
        if system_parts:
            config["system_instruction"] = "\n\n".join(system_parts)
        if self._settings.json_mode:
            config["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=self._settings.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config),
        )
        text = self._google_response_text(response)
        if not text.strip():
            raise ActionValidationError("Gemini returned an empty completion")
        return text

    @staticmethod
    def _google_response_text(response: Any) -> str:
        """Extract text from a Gemini response, tolerating a blocked candidate."""
        chunks: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        if chunks:
            return "".join(chunks)
        try:
            return str(getattr(response, "text", "") or "")
        except Exception:  # pragma: no cover - response.text raises when blocked
            return ""

    def _to_anthropic_messages(
        messages: Sequence[Mapping[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert OpenAI-style messages into Anthropic's content-block format."""
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                continue

            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, Sequence):
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": str(block.get("text", ""))})
                    elif block.get("type") == "image_url":
                        image_url = block.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, Mapping) else ""
                        if not url.startswith("data:"):
                            continue
                        header, _, data = url.partition(",")
                        media_type = header[5:].split(";")[0] or "image/png"
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )

            if not blocks:
                blocks.append({"type": "text", "text": " "})

            # Anthropic requires alternating roles; merge consecutive turns.
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        if not converted:
            converted = [{"role": "user", "content": [{"type": "text", "text": " "}]}]
        return "\n\n".join(system_parts), converted


__all__ = [
    "SYSTEM_PROMPT",
    "MemoryEntry",
    "ScreenCapture",
    "ScreenFrame",
    "ScreenGeometry",
    "ScreenHistory",
    "StateVisit",
    "TrajectoryMemory",
    "VisionLLM",
    "describe_environment",
]
