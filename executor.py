"""Actuation layer: validated VLM actions -> safe OS-level driver calls.

Three responsibilities, deliberately separated from perception and planning:

1. :class:`CoordinateMapper` translates *screenshot pixels* into *logical desktop
   points* (handling downscaling, ROI offsets and Retina/HiDPI multipliers) and
   clamps the result away from display edges.
2. :class:`ActionExecutor` performs the action with humanized timing.
3. :class:`ExecutionResult` reports what happened so the agent can evaluate the
   step and feed the outcome back into trajectory memory.
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pyautogui

from config import AppConfig, HumanizationSettings, LoopSettings
from schema import (
    COORDINATE_SPACE,
    ActionExecutionError,
    ActionType,
    AgentAction,
    Point,
    SafetyAbort,
)
from vision_llm import ScreenFrame, ScreenGeometry

LOGGER = logging.getLogger(__name__)


#: Friendly key names the VLM may emit, mapped to PyAutoGUI's vocabulary.
_KEY_ALIASES: dict[str, str] = {
    "win": "win",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "command": "win",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "pgdn": "pagedown",
    "page_up": "pageup",
    "page_down": "pagedown",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "bksp": "backspace",
    "spacebar": "space",
    "prtsc": "printscreen",
    "print": "printscreen",
}


def _resolve_key_chord(key_input: Optional[str]) -> list[str]:
    """Turn ``ctrl+s`` into PyAutoGUI key names, rejecting unknown keys."""
    if not key_input or not key_input.strip():
        raise ActionExecutionError("PRESS_KEY is missing key_input")

    known = {name.lower() for name in pyautogui.KEYBOARD_KEYS}
    chord: list[str] = []
    for token in key_input.strip().lower().replace(" ", "").split("+"):
        key = _KEY_ALIASES.get(token, token)
        if key not in known:
            raise ActionExecutionError(
                f"unknown key {token!r}; use names like win, enter, esc, tab, space, "
                "backspace, delete, up/down/left/right, f1-f12, or a chord such as ctrl+s"
            )
        chord.append(key)
    if not chord:
        raise ActionExecutionError("PRESS_KEY is missing key_input")
    return chord


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalBounds:
    """Clampable desktop rectangle expressed in PyAutoGUI coordinates."""

    left: int
    top: int
    width: int
    height: int

    @classmethod
    def from_geometry(cls, geometry: ScreenGeometry) -> "LogicalBounds":
        left, top, width, height = geometry.logical_bounds
        return cls(left=left, top=top, width=width, height=height)

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    def clamp(self, x: int, y: int, buffer: int) -> tuple[int, int]:
        """Keep a point ``buffer`` pixels inside the rectangle.

        The buffer prevents the cursor from parking on an edge monitor seam or
        triggering OS corner gestures (macOS Hot Corners, Windows edge swipes).
        """
        pad = max(0, int(buffer))
        min_x, max_x = self.left + pad, self.right - pad
        min_y, max_y = self.top + pad, self.bottom - pad
        if min_x > max_x:
            min_x = max_x = self.left + self.width // 2
        if min_y > max_y:
            min_y = max_y = self.top + self.height // 2
        return (min(max(int(x), min_x), max_x), min(max(int(y), min_y), max_y))


class CoordinateMapper:
    """Map coordinates from the *sent image* to the *logical desktop*."""

    def __init__(self, frame: ScreenFrame, bounds: LogicalBounds, buffer: int) -> None:
        if frame.sent_width <= 0 or frame.sent_height <= 0:
            raise ValueError("frame has an invalid size")
        if frame.captured_width <= 0 or frame.captured_height <= 0:
            raise ValueError("frame capture has an invalid size")
        if frame.scale <= 0:
            raise ValueError("frame scale must be positive")
        self._frame = frame
        self._bounds = bounds
        self._buffer = buffer
        self._x_ratio = frame.captured_width / frame.sent_width
        self._y_ratio = frame.captured_height / frame.sent_height

    def to_logical(self, point: Point) -> tuple[int, int]:
        """Convert one normalized (0-1000) point to a clamped logical coordinate."""
        norm_x, norm_y = point
        # 0. Model grid -> pixels of the image the model actually saw.
        image_x = norm_x / COORDINATE_SPACE * self._frame.sent_width
        image_y = norm_y / COORDINATE_SPACE * self._frame.sent_height
        # 1. Undo downscaling -> physical pixels inside the captured region.
        physical_x = self._frame.origin_x + image_x * self._x_ratio
        physical_y = self._frame.origin_y + image_y * self._y_ratio
        # 2. Physical pixels -> logical points via the DPI multiplier.
        logical_x = int(round(physical_x / self._frame.scale))
        logical_y = int(round(physical_y / self._frame.scale))
        # 3. Keep away from the absolute screen boundaries.
        clamped_x, clamped_y = self._bounds.clamp(logical_x, logical_y, self._buffer)

        if (clamped_x, clamped_y) != (logical_x, logical_y):
            LOGGER.warning(
                "Coordinate clamped: image(%d, %d) -> logical(%d, %d) -> (%d, %d) "
                "[buffer=%d, bounds=%s]",
                image_x,
                image_y,
                logical_x,
                logical_y,
                clamped_x,
                clamped_y,
                self._buffer,
                self._bounds,
            )
        else:
            LOGGER.debug(
                "Coordinate mapped: image(%d, %d) x%.3f -> physical(%d, %d) /%.3f -> logical(%d, %d)",
                image_x,
                image_y,
                self._x_ratio,
                physical_x,
                physical_y,
                self._frame.scale,
                logical_x,
                logical_y,
            )
        return clamped_x, clamped_y


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of a single actuator call."""

    action: AgentAction
    success: bool
    detail: str
    logical_coordinates: tuple[Point, ...] = ()
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ActionExecutor:
    """Drive PyAutoGUI with clamped coordinates and randomized timing."""

    def __init__(self, config: AppConfig, bounds: LogicalBounds) -> None:
        self._config = config
        self._bounds = bounds
        self._human: HumanizationSettings = config.humanization
        self._loop: LoopSettings = config.loop

        # Global PyAutoGUI policy. FAILSAFE lets the human slam the cursor into a
        # screen corner to abort; PAUSE is disabled because every delay here is
        # explicit and randomized.
        pyautogui.FAILSAFE = config.safety.failsafe
        pyautogui.PAUSE = 0.0
        LOGGER.info(
            "Executor ready: failsafe=%s bounds=%s buffer=%dpx",
            pyautogui.FAILSAFE,
            bounds,
            config.display.boundary_buffer,
        )

    # -- public API ---------------------------------------------------------

    def execute(self, action: AgentAction, frame: ScreenFrame) -> ExecutionResult:
        """Execute one validated action. Raises on driver or safety failure."""
        mapper = CoordinateMapper(frame, self._bounds, self._config.display.boundary_buffer)
        started = time.monotonic()
        handler = self._handlers()[action.action_type]

        LOGGER.info("ACT %s", action.describe())
        try:
            detail, points = handler(action, mapper)
        except pyautogui.FailSafeException as exc:
            raise SafetyAbort(
                "PyAutoGUI fail-safe triggered (cursor moved to a screen corner). "
                "The run has been stopped."
            ) from exc
        except (ActionExecutionError, SafetyAbort):
            raise
        except Exception as exc:
            raise ActionExecutionError(
                f"{action.action_type.value} failed: {type(exc).__name__}: {exc}"
            ) from exc

        elapsed = time.monotonic() - started
        LOGGER.info("ACT complete in %.2fs: %s", elapsed, detail)
        return ExecutionResult(
            action=action,
            success=True,
            detail=detail,
            logical_coordinates=points,
            elapsed_seconds=elapsed,
        )

    # -- dispatch -----------------------------------------------------------

    def _handlers(
        self,
    ) -> dict[ActionType, Callable[[AgentAction, CoordinateMapper], tuple[str, tuple[Point, ...]]]]:
        return {
            ActionType.CLICK: self._click,
            ActionType.DOUBLE_CLICK: self._double_click,
            ActionType.TYPE: self._type,
            ActionType.PRESS_KEY: self._press_key,
            ActionType.SCROLL: self._scroll,
            ActionType.DRAG: self._drag,
            ActionType.WAIT: self._wait,
            ActionType.DONE: self._done,
            ActionType.ESCALATE_TO_HUMAN: self._escalate,
        }

    # -- actions ------------------------------------------------------------

    def _click(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        if not action.coordinates:
            raise ActionExecutionError("CLICK requires one coordinate")
        target = mapper.to_logical(action.coordinates[0])
        self._move_bezier(target)
        self._sleep_range(self._human.pre_click_delay_range)
        pyautogui.click(target[0], target[1], button="left")
        LOGGER.info("Clicked left button at screen%s", target)
        return f"clicked at {target}", (target,)

    def _double_click(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        if not action.coordinates:
            raise ActionExecutionError("DOUBLE_CLICK requires one coordinate")
        target = mapper.to_logical(action.coordinates[0])
        self._move_bezier(target)
        self._sleep_range(self._human.pre_click_delay_range)
        interval = random.uniform(0.08, 0.18)
        pyautogui.doubleClick(target[0], target[1], button="left", interval=interval)
        LOGGER.info("Double-clicked at screen%s (interval=%.3fs)", target, interval)
        return f"double-clicked at {target}", (target,)

    def _type(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        text = action.text_input
        if text is None:
            raise ActionExecutionError("TYPE action is missing text_input")
        points: tuple[Point, ...] = ()
        if action.coordinates:
            target = mapper.to_logical(action.coordinates[0])
            self._move_bezier(target)
            self._sleep_range(self._human.pre_click_delay_range)
            pyautogui.click(target[0], target[1], button="left")
            points = (target,)
            time.sleep(random.uniform(0.10, 0.22))  # let the field take focus

        self._type_text(text)
        LOGGER.info(
            "Typed %d character(s)%s",
            len(text),
            f" into focus at screen{points[0]}" if points else " into the focused control",
        )
        return f"typed {len(text)} character(s)", points

    def _press_key(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        chord = _resolve_key_chord(action.key_input)
        self._sleep_range(self._human.pre_click_delay_range)
        if len(chord) == 1:
            pyautogui.press(chord[0])
            LOGGER.info("Pressed key %s", chord[0])
            return f"pressed {chord[0]}", ()
        pyautogui.hotkey(*chord)
        LOGGER.info("Pressed key chord %s", "+".join(chord))
        return f"pressed {'+'.join(chord)}", ()

    def _scroll(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        points: tuple[Point, ...] = ()
        if action.coordinates:
            target = mapper.to_logical(action.coordinates[0])
            self._move_bezier(target)
            self._sleep_range(self._human.pre_click_delay_range)
            points = (target,)

        clicks = int(action.scroll_amount or self._loop.scroll_clicks_default)
        clicks = max(-self._loop.scroll_clicks_max, min(self._loop.scroll_clicks_max, clicks))
        pyautogui.scroll(clicks)
        LOGGER.info("Scrolled %+d wheel click(s)%s", clicks, f" at screen{points[0]}" if points else "")
        return f"scrolled {clicks:+d}", points

    def _drag(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        if not action.coordinates or len(action.coordinates) != 2:
            raise ActionExecutionError("DRAG requires exactly two coordinates")
        start = mapper.to_logical(action.coordinates[0])
        end = mapper.to_logical(action.coordinates[1])

        self._move_bezier(start)
        self._sleep_range(self._human.pre_click_delay_range)
        pyautogui.mouseDown(start[0], start[1], button="left")
        try:
            time.sleep(random.uniform(0.12, 0.25))
            # A drag should be slower and smoother than a free mouse move.
            duration = random.uniform(*self._human.move_duration_range) * 1.6
            self._move_bezier(end, duration=duration, hold_button=True)
            time.sleep(random.uniform(0.08, 0.16))
        finally:
            pyautogui.mouseUp(button="left")
        LOGGER.info("Dragged screen%s -> screen%s", start, end)
        return f"dragged {start} -> {end}", (start, end)

    def _wait(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        seconds = random.uniform(*self._human.wait_action_range)
        LOGGER.info("Waiting %.2fs for the UI to settle", seconds)
        time.sleep(seconds)
        return f"waited {seconds:.2f}s", ()

    def _done(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        LOGGER.info("Model reported DONE: %s", action.thought_process)
        return "done", ()

    def _escalate(
        self, action: AgentAction, mapper: CoordinateMapper
    ) -> tuple[str, tuple[Point, ...]]:
        reason = action.text_input or action.thought_process
        LOGGER.warning("Model requested human escalation: %s", reason)
        return f"escalated: {reason}", ()

    # -- humanized primitives ----------------------------------------------

    @staticmethod
    def _sleep_range(bounds: tuple[float, float]) -> None:
        low, high = bounds
        time.sleep(random.uniform(low, high))

    def _move_bezier(
        self,
        target: tuple[int, int],
        *,
        duration: Optional[float] = None,
        hold_button: bool = False,
    ) -> None:
        """Glide the cursor along a jittered cubic Bezier curve.

        Straight-line ``moveTo`` is a strong automation fingerprint. Two random
        control points bowed off the direct path, plus per-step timing jitter,
        produce natural-looking movement while still landing exactly on target.
        """
        start_x, start_y = pyautogui.position()
        dx, dy = target[0] - start_x, target[1] - start_y
        distance = math.hypot(dx, dy)

        if distance < 2:
            pyautogui.moveTo(target[0], target[1], duration=0)
            return

        total_duration = (
            duration
            if duration is not None
            else random.uniform(*self._human.move_duration_range)
        )
        steps = max(8, min(64, int(distance / 16) + 8))
        jitter = max(0.0, float(self._human.move_jitter_pixels))

        # Perpendicular bow for the two interior control points.
        perp_x, perp_y = -dy / distance, dx / distance
        bow_one = random.uniform(-0.18, 0.18) * distance
        bow_two = random.uniform(-0.18, 0.18) * distance
        p0 = (float(start_x), float(start_y))
        p1 = (start_x + dx * 0.33 + perp_x * bow_one, start_y + dy * 0.33 + perp_y * bow_one)
        p2 = (start_x + dx * 0.66 + perp_x * bow_two, start_y + dy * 0.66 + perp_y * bow_two)
        p3 = (float(target[0]), float(target[1]))

        step_delay = total_duration / steps
        for index in range(1, steps + 1):
            t = index / steps
            eased = t * t * (3.0 - 2.0 * t)  # smoothstep for acceleration/deceleration
            x = self._cubic(p0[0], p1[0], p2[0], p3[0], eased)
            y = self._cubic(p0[1], p1[1], p2[1], p3[1], eased)
            if index < steps and jitter:
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            pyautogui.moveTo(round(x), round(y), duration=0)
            time.sleep(step_delay * random.uniform(0.7, 1.3))

        # Guarantee we finish exactly on target despite jitter.
        pyautogui.moveTo(target[0], target[1], duration=0)
        LOGGER.debug(
            "Mouse path: start=(%d, %d) target=%s steps=%d duration=%.2fs hold=%s",
            start_x,
            start_y,
            target,
            steps,
            total_duration,
            hold_button,
        )

    @staticmethod
    def _cubic(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
        inv = 1.0 - t
        return (inv**3) * p0 + 3 * (inv**2) * t * p1 + 3 * inv * (t**2) * p2 + (t**3) * p3

    def _type_text(self, text: str) -> None:
        """Type character by character with per-keystroke jitter.

        Raw text is only logged at DEBUG: it may contain sensitive data, and the
        console log should stay shareable. Each keystroke is still auditable when
        ``AGENT_LOG_LEVEL=DEBUG``.
        """
        if not text:
            return
        if any(ord(char) > 127 for char in text):
            self._paste_text(text)
            return

        delay_range = self._human.key_delay_range
        for char in text:
            if char == "\n":
                pyautogui.press("enter")
            elif char == "\t":
                pyautogui.press("tab")
            else:
                pyautogui.write(char)
            LOGGER.debug("Keystroke: %r", char)
            time.sleep(random.uniform(*delay_range))

    def _paste_text(self, text: str) -> None:
        """Fallback for non-ASCII text (emoji, accents, CJK) via the clipboard."""
        try:
            import pyperclip  # optional; ships with pyautogui's dependency tree
        except Exception:
            pyperclip = None  # type: ignore[assignment]

        if pyperclip is None:
            raise ActionExecutionError(
                "Typed text contains non-ASCII characters but pyperclip is not "
                "installed. Install pyperclip or restrict text to ASCII."
            )

        previous: Optional[str]
        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None

        try:
            pyperclip.copy(text)
            modifier = "command" if sys.platform == "darwin" else "ctrl"
            time.sleep(0.08)
            pyautogui.hotkey(modifier, "v")
            LOGGER.debug("Pasted %d non-ASCII character(s) via clipboard", len(text))
            time.sleep(random.uniform(0.15, 0.30))
        finally:
            if previous is not None:
                try:
                    pyperclip.copy(previous)
                except Exception:
                    LOGGER.debug("Could not restore clipboard contents", exc_info=True)
