"""Main OODA driver: Observe -> Orient -> Decide -> Act -> Evaluate.

``DesktopAgent`` owns the loop, the global kill-switch, human-in-the-loop
confirmation, stagnation detection, and the audit transcript for a run.
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pyautogui
from PIL import Image, ImageChops, ImageFilter

from config import AppConfig, DisplaySettings, attach_file_handler
from executor import ActionExecutor, ExecutionResult, LogicalBounds
from schema import (
    ActionExecutionError,
    ActionType,
    ActionValidationError,
    AgentAction,
    CaptureError,
    ConfigurationError,
    SafetyAbort,
    StagnationError,
)
from vision_llm import (
    ScreenCapture,
    ScreenFrame,
    ScreenHistory,
    StateVisit,
    TrajectoryMemory,
    VisionLLM,
)

try:  # Global hotkeys are unavailable on some headless/CI environments.
    import keyboard
except Exception:  # pragma: no cover - exercised only when the package is missing
    keyboard = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

#: Thumbnail size used for frame-to-frame comparison (cheap, still sensitive
#: enough to catch a single typed line).
THUMBNAIL_SIZE: tuple[int, int] = (160, 90)
#: Frames sampled while idle to learn which pixels animate on their own (text
#: carets, spinners, clock digits). They are ignored when judging an action.
IDLE_MASK_FRAMES: int = 3
IDLE_MASK_DELAY: float = 0.35
#: Dilation (in thumbnail pixels) applied to the idle mask so a caret blinking a
#: pixel to the side is still covered.
IDLE_MASK_GROW: int = 3


def thumbnail(image: Image.Image) -> Image.Image:
    """Reduce a screenshot to the grayscale thumbnail used for change detection."""
    return image.convert("L").resize(THUMBNAIL_SIZE, Image.Resampling.BILINEAR)


def build_idle_mask(
    thumbnails: list[Image.Image], settings: DisplaySettings
) -> Optional[Image.Image]:
    """Pixels that change on their own while nothing is being done.

    Two consecutive idle frames mark a caret or spinner exactly where it blinks.
    Masking those pixels keeps a blinking cursor from being mistaken for a real
    update, which lets the sensitivity threshold stay low enough to notice a
    single typed line.
    """
    if len(thumbnails) < 2:
        return None
    try:
        animated = Image.new("L", thumbnails[0].size, 0)
        for before, after in zip(thumbnails, thumbnails[1:]):
            difference = ImageChops.difference(before, after)
            hot = difference.point(lambda v: 255 if v >= settings.screen_change_pixel_delta else 0)
            animated = ImageChops.lighter(animated, hot)
        if IDLE_MASK_GROW > 0:
            animated = animated.filter(ImageFilter.MaxFilter(IDLE_MASK_GROW * 2 + 1))
        return animated
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Idle mask could not be built", exc_info=True)
        return None


def compare_frames(
    previous: Image.Image,
    current: Image.Image,
    settings: DisplaySettings,
    idle_mask: Optional[Image.Image] = None,
) -> tuple[bool, float, float]:
    """Return ``(changed, mean_delta, changed_fraction)`` for two thumbnails.

    A whole-frame brightness average is blind to small, real updates: one typed
    line changes about 0.3% of the screen. So the frame also counts as changed
    when enough pixels differ by ``screen_change_pixel_delta`` or more, ignoring
    the pixels known to animate on their own.
    """
    try:
        difference = ImageChops.difference(previous, current)
        histogram = difference.histogram()
        hot = difference.point(
            lambda v: 255 if v >= settings.screen_change_pixel_delta else 0
        )
        if idle_mask is not None:
            hot = ImageChops.subtract(hot, idle_mask)
        hot_count = sum(hot.histogram()[128:])
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Screen comparison failed", exc_info=True)
        return True, 0.0, 0.0

    total = sum(histogram) or 1
    mean_delta = sum(value * count for value, count in enumerate(histogram)) / total
    fraction = hot_count / total
    changed = (
        mean_delta >= settings.screen_change_threshold
        or fraction >= settings.screen_change_fraction
    )
    return changed, mean_delta, fraction


class RunStatus(str, Enum):
    """Terminal state of a single :meth:`DesktopAgent.run` call."""

    DONE = "DONE"
    ESCALATED = "ESCALATED_TO_HUMAN"
    STAGNATED = "STAGNATED"
    MAX_STEPS = "MAX_STEPS_REACHED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AgentResult:
    """Everything the caller (and the transcript) needs to know about a run."""

    status: RunStatus
    message: str
    steps: int
    elapsed_seconds: float
    run_dir: Optional[Path] = None

    @property
    def succeeded(self) -> bool:
        return self.status is RunStatus.DONE


class DesktopAgent:
    """Stateful orchestrator for one goal."""

    def __init__(
        self,
        config: AppConfig,
        *,
        capture: Optional[ScreenCapture] = None,
        memory: Optional[TrajectoryMemory] = None,
        llm: Optional[VisionLLM] = None,
        executor: Optional[ActionExecutor] = None,
    ) -> None:
        self._config = config
        self._capture = capture if capture is not None else ScreenCapture(config.display)
        self._memory = memory or TrajectoryMemory(
            maxlen=max(8, config.safety.max_identical_actions)
        )
        self._llm = llm if llm is not None else VisionLLM(config.llm, self._memory)
        self._bounds = LogicalBounds.from_geometry(self._capture.geometry)
        self._executor = executor if executor is not None else ActionExecutor(config, self._bounds)

        self._stop_event = threading.Event()
        self._kill_registered = False
        self._transcript: list[dict[str, Any]] = []
        self._run_dir: Optional[Path] = None
        self._previous_thumbnail: Optional[Image.Image] = None
        self._idle_mask: Optional[Image.Image] = None
        self._screen_history = ScreenHistory()
        # signature -> how many times it was tried without changing the screen
        self._futile_attempts: dict[str, int] = {}
        self._log_handler: Optional[logging.Handler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, goal: str) -> AgentResult:
        """Execute the OODA loop until done, blocked, or out of budget."""
        if not goal or not goal.strip():
            raise ValueError("A non-empty goal is required.")
        self._config.require_api_key()
        started = time.monotonic()
        # A second run on the same instance must not inherit the first run's
        # visited screens, frame history or per-signature failure counts.
        self._screen_history = ScreenHistory()
        self._previous_thumbnail = None
        self._futile_attempts.clear()
        self._memory.reset()
        self._prepare_run()
        self._idle_mask = self._learn_idle_mask()
        self._install_kill_switch()
        self._install_signal_handlers()

        LOGGER.info("=" * 78)
        LOGGER.info("DesktopAgent run started")
        LOGGER.info("Goal: %s", goal.strip())
        LOGGER.info(
            "Mode: hitl=%s failsafe=%s roi=%s max_steps=%d budget=%.0fs",
            self._config.safety.hitl,
            self._config.safety.failsafe,
            self._capture.geometry.roi or "full monitor",
            self._config.loop.max_steps,
            self._config.loop.max_runtime_seconds,
        )
        LOGGER.info("=" * 78)

        status = RunStatus.ERROR
        message = "Run ended without a result."
        step = 0

        try:
            while step < self._config.loop.max_steps:
                step += 1
                if self._stop_event.is_set():
                    status, message = RunStatus.ABORTED, "Stopped by user signal."
                    break

                elapsed = time.monotonic() - started
                if elapsed > self._config.loop.max_runtime_seconds:
                    status = RunStatus.ESCALATED
                    message = (
                        f"Runtime budget of {self._config.loop.max_runtime_seconds:.0f}s "
                        "exhausted before the goal completed."
                    )
                    self._escalate_to_human(message)
                    break

                # -- OBSERVE ------------------------------------------------
                frame = self._observe()
                LOGGER.info(
                    "OBSERVE step=%d/%d frame=%s",
                    step,
                    self._config.loop.max_steps,
                    frame.prompt_summary(),
                )
                self._save_screenshot(frame, step)

                # -- ORIENT --------------------------------------------------
                visit: StateVisit = self._screen_history.observe(
                    step, self._thumbnail(frame.image)
                )
                state_note = self._screen_history.prompt_block(visit)
                if state_note:
                    LOGGER.info(
                        "VISITED step=%d: telling the model what it already tried here "
                        "(%d attempt(s) from steps %s)",
                        step,
                        len(visit.attempts),
                        ", ".join(str(s) for s in visit.steps[:-1][-4:]),
                    )
                LOGGER.info(
                    "ORIENT step=%d history=%d screen=%s recent=%s",
                    step,
                    len(self._memory),
                    self._screen_history.describe_visit(visit),
                    [entry.action.action_type.value for entry in self._memory.entries[-3:]],
                )

                # -- DECIDE --------------------------------------------------
                limit = self._config.safety.max_state_repeats
                action: Optional[AgentAction] = None
                for decision_attempt in (1, 2):
                    action = self._llm.decide(frame, goal, step, state_note)
                    LOGGER.info("REASONING step=%d thought=%s", step, action.thought_process)
                    if action.action_type in (ActionType.DONE, ActionType.ESCALATE_TO_HUMAN):
                        break
                    repeats = len(self._screen_history.prior_attempts(visit, action))
                    if repeats < limit:
                        break
                    LOGGER.warning(
                        "Repeat rejected: %s was already tried %d time(s) from this same "
                        "screen (steps %s) with the same result.",
                        action.describe(),
                        repeats,
                        ", ".join(str(a.step) for a in self._screen_history.prior_attempts(visit, action)),
                    )
                    if decision_attempt == 2:
                        raise StagnationError(
                            f"The agent is looping: {action.describe()} has been attempted "
                            f"{repeats} time(s) from the same screen and keeps arriving at "
                            "the same place. A human needs to choose a different target."
                        )
                    state_note = "\n\n".join(
                        part
                        for part in (
                            state_note,
                            "## YOUR LAST ANSWER WAS REJECTED\n"
                            f"{action.describe()} was rejected: you already took exactly this "
                            f"action from this same screen {repeats} time(s), and it led to the "
                            "same place each time. Choose a visibly different target or a "
                            "different kind of action.",
                        )
                        if part
                    )
                assert action is not None  # the loop always assigns or raises
                # This thought describes what the previous action produced.
                self._screen_history.note_thought(action.thought_process)

                if action.action_type is ActionType.DONE:
                    status = RunStatus.DONE
                    message = action.thought_process
                    self._record_decision(step, action, "goal reported complete")
                    LOGGER.info("DONE: %s", action.thought_process)
                    break

                if action.action_type is ActionType.ESCALATE_TO_HUMAN:
                    status = RunStatus.ESCALATED
                    message = (
                        action.text_input or action.thought_process or "Model requested human help."
                    )
                    self._record_decision(step, action, "escalated before execution")
                    self._escalate_to_human(message)
                    break

                # -- HUMAN-IN-THE-LOOP GATE ---------------------------------
                if not self._confirm(action):
                    status = RunStatus.ABORTED
                    message = "Human declined the proposed action."
                    self._record_decision(step, action, "declined by human")
                    break

                futile = self._futile_attempts.get(action.signature(), 0)
                if futile >= self._config.safety.max_futile_attempts:
                    raise StagnationError(
                        f"{action.describe()} was already tried {futile} time(s) with no "
                        "visible change; the same action will not be repeated."
                    )

                # -- ACT -----------------------------------------------------
                self._screen_history.record_action(visit, step, action)
                try:
                    result = self._executor.execute(action, frame)
                except ActionExecutionError as exc:
                    LOGGER.error("ACT failed: %s", exc)
                    self._memory.record(step, action, f"FAILED: {exc}")
                    self._record_decision(step, action, f"FAILED: {exc}")
                    failures = self._consecutive_failures()
                    if failures >= self._config.safety.max_consecutive_failures:
                        status = RunStatus.ESCALATED
                        message = (
                            f"{failures} consecutive execution failures; "
                            f"last error: {exc}"
                        )
                        self._escalate_to_human(message)
                        break
                    # Brief pause so a broken driver is not hammered.
                    time.sleep(random.uniform(*self._config.humanization.post_action_settle_range))
                    continue

                self._memory.record(step, action, result.detail)
                self._record_decision(step, action, result.detail)
                LOGGER.info(
                    "EVALUATE step=%d executed=%s logical=%s in %.2fs",
                    step,
                    action.action_type.value,
                    list(result.logical_coordinates) or "n/a",
                    result.elapsed_seconds,
                )

                # -- stagnation cap -----------------------------------------
                if self._memory.is_stagnant(self._config.safety.max_identical_actions):
                    raise StagnationError(
                        "The last "
                        f"{self._config.safety.max_identical_actions} actions were identical "
                        f"({action.describe()}); the UI is not progressing."
                    )

                time.sleep(random.uniform(*self._config.humanization.post_action_settle_range))
            else:
                status = RunStatus.MAX_STEPS
                message = (
                    f"Step budget of {self._config.loop.max_steps} exhausted before the goal "
                    "was reported complete."
                )
                self._escalate_to_human(message)

        except StagnationError as exc:
            status = RunStatus.STAGNATED
            message = str(exc)
            LOGGER.critical("Stagnation detected: %s", exc)
            self._escalate_to_human(str(exc))
        except SafetyAbort as exc:
            status, message = RunStatus.ABORTED, str(exc)
            LOGGER.critical("Safety abort: %s", exc)
        except CaptureError as exc:
            status, message = RunStatus.ERROR, f"Screen capture failed: {exc}"
            LOGGER.error(message)
        except ActionValidationError as exc:
            status, message = RunStatus.ERROR, f"Invalid model output: {exc}"
            LOGGER.error(message)
        except ConfigurationError as exc:
            status, message = RunStatus.ERROR, str(exc)
            LOGGER.error("Configuration error: %s", exc)
        except pyautogui.FailSafeException as exc:
            status, message = RunStatus.ABORTED, "PyAutoGUI fail-safe triggered."
            LOGGER.critical("%s", message, exc_info=exc)
        except Exception as exc:  # noqa: BLE001 - top-level guard for a long run
            status, message = RunStatus.ERROR, f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Unhandled agent failure")
        finally:
            elapsed = time.monotonic() - started
            self._cleanup()
            self._write_transcript(goal, status, message, step, elapsed)

        result = AgentResult(
            status=status,
            message=message,
            steps=step,
            elapsed_seconds=time.monotonic() - started,
            run_dir=self._run_dir,
        )
        LOGGER.info(
            "Run finished: status=%s steps=%d elapsed=%.1fs message=%s",
            result.status.value,
            result.steps,
            result.elapsed_seconds,
            result.message,
        )
        return result

    # ------------------------------------------------------------------
    # Safety mechanisms
    # ------------------------------------------------------------------

    def _install_kill_switch(self) -> None:
        settings = self._config.safety
        if not settings.kill_switch_enabled:
            LOGGER.warning("Global kill-switch is DISABLED by configuration.")
            return
        if keyboard is None:
            LOGGER.warning(
                "The 'keyboard' package is unavailable; global kill-switch %s is not armed. "
                "PyAutoGUI fail-safe remains active.",
                settings.kill_switch_hotkey,
            )
            return
        try:
            keyboard.add_hotkey(settings.kill_switch_hotkey, self._handle_kill_switch, suppress=False)
            self._kill_registered = True
            LOGGER.info(
                "Global kill-switch armed: %s (terminates the process immediately)",
                settings.kill_switch_hotkey,
            )
        except Exception as exc:  # pragma: no cover - OS policy specific
            LOGGER.error(
                "Could not register kill-switch %s (%s). On Linux/macOS this often "
                "requires elevated permissions; PyAutoGUI fail-safe is still active.",
                settings.kill_switch_hotkey,
                exc,
            )

    def _handle_kill_switch(self) -> None:
        hotkey = self._config.safety.kill_switch_hotkey
        LOGGER.critical("!!! KILL-SWITCH %s ACTIVATED - TERMINATING NOW !!!", hotkey)
        try:
            logging.shutdown()
        finally:
            os._exit(130)

    def _install_signal_handlers(self) -> None:
        """Turn SIGINT/SIGTERM into a graceful stop between steps."""
        for signal_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, signal_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):  # not the main thread / unsupported platform
                LOGGER.debug("Could not install %s handler", signal_name)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        LOGGER.warning("Signal %s received: finishing the current step, then stopping.", signum)
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _confirm(self, action: AgentAction) -> bool:
        """HITL gate. Returns True when the action may be executed."""
        if not self._config.safety.hitl:
            return True

        print("\n" + "-" * 70)
        print("HUMAN-IN-THE-LOOP CONFIRMATION")
        print(f"Thought : {action.thought_process}")
        print(f"Action  : {action.describe()}")
        print("-" * 70)
        LOGGER.info("HITL prompt for action=%s", action.describe())

        while True:
            try:
                answer = input("Execute this action? [y]es / [n]o / [q]uit: ").strip().lower()
            except EOFError:
                LOGGER.warning("HITL input closed (EOF); declining the action.")
                return False
            except KeyboardInterrupt:
                LOGGER.warning("Interrupted at the HITL prompt; aborting the run.")
                self._stop_event.set()
                return False

            if answer in {"y", "yes"}:
                LOGGER.info("HITL approved action=%s", action.action_type.value)
                return True
            if answer in {"n", "no"}:
                LOGGER.warning("HITL declined action=%s", action.action_type.value)
                return False
            if answer in {"q", "quit"}:
                LOGGER.warning("HITL requested quit.")
                self._stop_event.set()
                return False
            print("Please answer 'y', 'n' or 'q'.")

    def _observe(self) -> ScreenFrame:
        """Grab a frame and decide whether the previous action changed anything.

        A full-frame brightness average hides small but real updates (one typed
        line moves roughly 0.3% of the pixels), and a slow page can repaint only
        after the first grab. So a change is either a large average shift or a
        small share of clearly different pixels, and an apparently unchanged
        screen is re-checked briefly before the loop calls the action a no-op.
        """
        settings = self._config.display
        frame = self._capture.grab()
        changed, mean_delta, fraction = self._compare_to_previous(frame.image)

        for attempt in range(1, settings.screen_change_rechecks + 1):
            if changed:
                break
            time.sleep(settings.screen_change_recheck_delay)
            candidate = self._capture.grab()
            again, again_mean, again_fraction = self._compare_to_previous(candidate.image)
            LOGGER.info(
                "REOBSERVE attempt=%d/%d screen_delta=%.2f changed_fraction=%.3f%% changed=%s",
                attempt,
                settings.screen_change_rechecks,
                again_mean,
                again_fraction * 100,
                again,
            )
            if again:
                frame, changed, mean_delta, fraction = candidate, again, again_mean, again_fraction
                break

        self._previous_thumbnail = self._thumbnail(frame.image)
        self._memory.update_last_screen_change(changed)
        if changed:
            # Real progress: earlier failures are no longer evidence of futility.
            self._futile_attempts.clear()
        elif self._memory.entries:
            signature = self._memory.entries[-1].action.signature()
            self._futile_attempts[signature] = self._futile_attempts.get(signature, 0) + 1

        LOGGER.info(
            "EVALUATE screen_delta=%.2f changed_fraction=%.3f%% changed=%s "
            "(mean>=%.2f or fraction>=%.2f%%)",
            mean_delta,
            fraction * 100,
            changed,
            settings.screen_change_threshold,
            settings.screen_change_fraction * 100,
        )
        if not changed and self._memory.entries:
            LOGGER.warning(
                "Screen appears unchanged since the previous action (%s).",
                self._memory.entries[-1].action.action_type.value,
            )
        return frame

    def _learn_idle_mask(self) -> Optional[Image.Image]:
        """Sample the untouched screen so self-animating pixels can be ignored."""
        frames: list[Image.Image] = []
        try:
            for index in range(IDLE_MASK_FRAMES):
                if index:
                    time.sleep(IDLE_MASK_DELAY)
                frames.append(self._thumbnail(self._capture.grab().image))
        except Exception:  # pragma: no cover - capture failure is handled elsewhere
            LOGGER.debug("Idle sampling failed", exc_info=True)
            return None
        mask = build_idle_mask(frames, self._config.display)
        if mask is not None:
            animated = sum(mask.histogram()[128:]) / (mask.width * mask.height)
            LOGGER.info(
                "Idle mask: ignoring %.2f%% of the screen that animates on its own "
                "(caret, spinners, clocks)",
                animated * 100,
            )
        return mask

    def _compare_to_previous(self, image: Image.Image) -> tuple[bool, float, float]:
        """Return ``(changed, mean_delta, changed_fraction)`` against the last frame."""
        if self._previous_thumbnail is None:
            return True, 0.0, 0.0
        return compare_frames(
            self._previous_thumbnail, self._thumbnail(image), self._config.display, self._idle_mask
        )

    @staticmethod
    def _thumbnail(image: Image.Image) -> Image.Image:
        return thumbnail(image)

    def _consecutive_failures(self) -> int:
        count = 0
        for entry in reversed(self._memory.entries):
            if entry.outcome.upper().startswith("FAILED"):
                count += 1
            else:
                break
        return count

    def _record_decision(self, step: int, action: AgentAction, outcome: str) -> None:
        self._transcript.append(
            {
                "step": step,
                "thought_process": action.thought_process,
                "action": action.action_type.value,
                "coordinates": action.coordinates,
                "text_input": action.text_input,
                "scroll_amount": action.scroll_amount,
                "outcome": outcome,
            }
        )

    def _escalate_to_human(self, reason: str) -> None:
        """Surface a blocking condition to the operator (log + optional alert)."""
        LOGGER.critical("=" * 78)
        LOGGER.critical("ESCALATE_TO_HUMAN: %s", reason)
        LOGGER.critical("=" * 78)
        if not self._config.safety.escalation_alert:
            return
        try:
            pyautogui.alert(text=reason, title="AutoClick needs a human")
        except Exception:  # pragma: no cover - no GUI session
            LOGGER.warning("Could not display the escalation alert dialog.", exc_info=True)

    # ------------------------------------------------------------------
    # Run artifacts
    # ------------------------------------------------------------------

    def _prepare_run(self) -> None:
        settings = self._config.display
        if not (settings.save_screenshots or self._config.logging.log_to_file):
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = settings.runs_dir / f"run-{stamp}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.warning("Could not create run directory %s: %s", run_dir, exc)
            return
        self._run_dir = run_dir
        if self._config.logging.log_to_file:
            self._log_handler = attach_file_handler(run_dir / "agent.log")
        LOGGER.info("Run artifacts: %s", run_dir)

    def _save_screenshot(self, frame: ScreenFrame, step: int) -> None:
        if self._run_dir is None or not self._config.display.save_screenshots:
            return
        path = self._run_dir / f"step_{step:03d}.png"
        try:
            self._capture.save_frame(frame, path)
        except Exception as exc:  # pragma: no cover - disk issues only
            LOGGER.warning("Could not save screenshot %s: %s", path, exc)

    def _cleanup(self) -> None:
        if self._kill_registered and keyboard is not None:
            hotkey = self._config.safety.kill_switch_hotkey
            try:
                keyboard.remove_hotkey(hotkey)
            except Exception:
                try:
                    keyboard.unhook_all_hotkeys()
                except Exception:
                    LOGGER.debug("Could not unhook global hotkeys", exc_info=True)
            self._kill_registered = False

        close_llm = getattr(self._llm, "close", None)
        if callable(close_llm):
            close_llm()
        close_capture = getattr(self._capture, "close", None)
        if callable(close_capture):
            close_capture()
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    def _write_transcript(
        self, goal: str, status: RunStatus, message: str, steps: int, elapsed: float
    ) -> None:
        if self._run_dir is None:
            return
        payload = {
            "goal": goal.strip(),
            "status": status.value,
            "message": message,
            "steps_executed": steps,
            "elapsed_seconds": round(elapsed, 2),
            "configuration": self._config.describe(),
            "actions": self._transcript,
        }
        path = self._run_dir / "transcript.json"
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            LOGGER.info("Transcript written to %s", path)
        except OSError as exc:  # pragma: no cover
            LOGGER.warning("Could not write transcript %s: %s", path, exc)


__all__ = ["AgentResult", "DesktopAgent", "RunStatus", "ExecutionResult", "ConfigurationError"]
