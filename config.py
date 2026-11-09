"""Central configuration for the AutoClick desktop agent.

Every tunable lives in this module. Values are resolved in this order:

1. Dataclass defaults below.
2. Environment variables (optionally loaded from a ``.env`` file).
3. Explicit CLI overrides applied in :mod:`main` via ``dataclasses.replace``.

Nothing in this module touches the screen or the network, so it is always safe
to import and to unit-test.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Literal, Optional, Tuple

from dotenv import load_dotenv

from schema import ConfigurationError

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"

# ``(left, top, width, height)`` in physical screen pixels (MSS coordinates).
Roi = Tuple[int, int, int, int]
# ``(minimum, maximum)`` inclusive seconds / pixels.
NumberRange = Tuple[float, float]

Provider = Literal["openai", "anthropic", "google"]


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a stripped, non-empty environment value or ``default``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


def _env_number_range(name: str, default: NumberRange) -> NumberRange:
    raw = _env(name)
    if raw is None:
        return default
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ConfigurationError(f"{name} must look like 'min,max', got {raw!r}")
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ConfigurationError(f"{name} must contain two numbers, got {raw!r}") from exc
    if low < 0 or high < low:
        raise ConfigurationError(f"{name} must satisfy 0 <= min <= max, got {raw!r}")
    return (low, high)


def parse_roi(raw: str) -> Roi:
    """Parse ``"left,top,width,height"`` into an :data:`Roi` tuple."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ConfigurationError(
            f"ROI must be 'left,top,width,height' (physical pixels), got {raw!r}"
        )
    try:
        left, top, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ConfigurationError(f"ROI values must be integers, got {raw!r}") from exc
    if width <= 0 or height <= 0:
        raise ConfigurationError(f"ROI width/height must be positive, got {raw!r}")
    return (left, top, width, height)


def _env_roi(name: str, default: Optional[Roi]) -> Optional[Roi]:
    raw = _env(name)
    if raw is None:
        return default
    return parse_roi(raw)


# ---------------------------------------------------------------------------
# Settings sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMSettings:
    """Vision-language-model transport and prompting settings."""

    provider: Provider = "openai"
    api_key: str = field(default="", repr=False)
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    request_timeout: float = 60.0
    # ``tenacity``: number of total attempts per HTTP request.
    api_max_attempts: int = 4
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 20.0
    # How many extra round-trips are allowed when the model returns bad JSON.
    schema_repair_attempts: int = 2
    # OpenAI-compatible ``response_format={"type": "json_object"}``. Disable for
    # gateways that reject the parameter.
    json_mode: bool = True

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = (_env("AGENT_PROVIDER", "google") or "google").lower()
        if provider not in {"openai", "anthropic", "google"}:
            raise ConfigurationError(
                f"AGENT_PROVIDER must be 'openai', 'anthropic', or 'google', got {provider!r}"
            )
        default_key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider)
        if default_key_var is None:
            raise ConfigurationError(f"Unsupported provider {provider!r}.")
        api_key = _env("AGENT_API_KEY", _env(default_key_var, "")) or ""
        model_default = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-5",
            "google": "gemini-2.5-flash",
        }.get(provider, "gpt-4o")
        model = _env("AGENT_MODEL", model_default)
        assert model is not None  # _env never returns None when a default is given
        return cls(
            provider=provider,  # type: ignore[arg-type]
            api_key=api_key,
            base_url=_env("AGENT_BASE_URL"),
            model=model,
            temperature=_env_float("AGENT_TEMPERATURE", 0.0),
            max_output_tokens=_env_int("AGENT_MAX_OUTPUT_TOKENS", 1024),
            request_timeout=_env_float("AGENT_REQUEST_TIMEOUT", 60.0),
            api_max_attempts=_env_int("AGENT_API_MAX_ATTEMPTS", 4),
            retry_initial_wait=_env_float("AGENT_RETRY_INITIAL_WAIT", 1.0),
            retry_max_wait=_env_float("AGENT_RETRY_MAX_WAIT", 20.0),
            schema_repair_attempts=_env_int("AGENT_SCHEMA_REPAIR_ATTEMPTS", 2),
            json_mode=_env_bool("AGENT_JSON_MODE", True),
        )


@dataclass(frozen=True)
class DisplaySettings:
    """Screen capture, ROI, downscaling, and DPI correction settings."""

    # MSS monitor index: 0 = all monitors, 1 = primary, 2.. = secondaries.
    monitor_index: int = 1
    # Optional crop, in *physical* pixels on the MSS virtual desktop.
    roi: Optional[Roi] = None
    # Physical-pixels-per-logical-point. ``None`` auto-detects from the primary
    # display (Windows scaling / macOS Retina). Override for mixed-DPI setups.
    display_scale: Optional[float] = None
    # Downscale the screenshot so its longest side is at most this many pixels.
    # Smaller values save tokens; coordinates are still relative to the image.
    max_image_side: int = 1600
    # Keep the cursor at least this many pixels away from every display edge.
    boundary_buffer: int = 5
    save_screenshots: bool = True
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    # Mean grayscale delta above which two frames count as "changed".
    screen_change_threshold: float = 1.5
    # A second, far more sensitive test: the share of pixels that must differ by
    # at least ``screen_change_pixel_delta`` before the screen counts as changed.
    # A full-frame average hides small but real updates such as one typed line
    # (about 0.3% of the screen), so both tests are applied and either wins.
    screen_change_fraction: float = 0.0005
    screen_change_pixel_delta: int = 8
    # Slow pages repaint after the first grab; re-check before concluding that an
    # action did nothing at all.
    screen_change_rechecks: int = 2
    screen_change_recheck_delay: float = 0.5

    @classmethod
    def from_env(cls) -> "DisplaySettings":
        raw_scale = _env("AGENT_DISPLAY_SCALE")
        return cls(
            monitor_index=_env_int("AGENT_MONITOR_INDEX", 1),
            roi=_env_roi("AGENT_ROI", None),
            display_scale=_env_float("AGENT_DISPLAY_SCALE", 1.0) if raw_scale else None,
            max_image_side=_env_int("AGENT_MAX_IMAGE_SIDE", 1600),
            boundary_buffer=_env_int("AGENT_BOUNDARY_BUFFER", 5),
            save_screenshots=_env_bool("AGENT_SAVE_SCREENSHOTS", True),
            runs_dir=Path(_env("AGENT_RUNS_DIR", str(PROJECT_ROOT / "runs")) or "runs"),
            screen_change_threshold=_env_float("AGENT_SCREEN_CHANGE_THRESHOLD", 1.5),
            screen_change_fraction=_env_float("AGENT_SCREEN_CHANGE_FRACTION", 0.0005),
            screen_change_pixel_delta=_env_int("AGENT_SCREEN_CHANGE_PIXEL_DELTA", 8),
            screen_change_rechecks=_env_int("AGENT_SCREEN_CHANGE_RECHECKS", 2),
            screen_change_recheck_delay=_env_float("AGENT_SCREEN_CHANGE_RECHECK_DELAY", 0.5),
        )


@dataclass(frozen=True)
class SafetySettings:
    """Human-in-the-loop, kill-switch, and loop safety limits."""

    hitl: bool = False
    kill_switch_enabled: bool = True
    kill_switch_hotkey: str = "ctrl+alt+k"
    failsafe: bool = True
    # Number of consecutive identical actions before StagnationError.
    max_identical_actions: int = 3
    # How many times the same action may be taken from the *same* screen before
    # it is rejected and the model is asked again. Catches A/B/A/B loops that
    # never produce two identical actions in a row.
    max_state_repeats: int = 2
    # How many times one identical action may be retried while the screen keeps
    # showing no change at all (catches click/escape ping-pong loops).
    max_futile_attempts: int = 2
    # Number of consecutive executor failures before escalating to the human.
    max_consecutive_failures: int = 3
    # Show a blocking OS alert when escalating. Off by default so unattended
    # runs never hang on a dialog.
    escalation_alert: bool = False

    @classmethod
    def from_env(cls) -> "SafetySettings":
        return cls(
            hitl=_env_bool("AGENT_HITL", False),
            kill_switch_enabled=_env_bool("AGENT_KILL_SWITCH_ENABLED", True),
            kill_switch_hotkey=_env("AGENT_KILL_SWITCH_HOTKEY", "ctrl+alt+k") or "ctrl+alt+k",
            failsafe=_env_bool("AGENT_FAILSAFE", True),
            max_identical_actions=_env_int("AGENT_MAX_IDENTICAL_ACTIONS", 3),
            max_state_repeats=_env_int("AGENT_MAX_STATE_REPEATS", 2),
            max_futile_attempts=_env_int("AGENT_MAX_FUTILE_ATTEMPTS", 2),
            max_consecutive_failures=_env_int("AGENT_MAX_CONSECUTIVE_FAILURES", 3),
            escalation_alert=_env_bool("AGENT_ESCALATION_ALERT", False),
        )


@dataclass(frozen=True)
class HumanizationSettings:
    """Randomized timing that makes synthetic input look like a human's."""

    key_delay_range: NumberRange = (0.05, 0.20)
    move_duration_range: NumberRange = (0.20, 0.55)
    move_jitter_pixels: float = 2.0
    pre_click_delay_range: NumberRange = (0.05, 0.20)
    post_action_settle_range: NumberRange = (0.35, 0.80)
    wait_action_range: NumberRange = (1.0, 2.0)

    @classmethod
    def from_env(cls) -> "HumanizationSettings":
        return cls(
            key_delay_range=_env_number_range("AGENT_KEY_DELAY_RANGE", (0.05, 0.20)),
            move_duration_range=_env_number_range("AGENT_MOVE_DURATION_RANGE", (0.20, 0.55)),
            move_jitter_pixels=_env_float("AGENT_MOVE_JITTER_PX", 2.0),
            pre_click_delay_range=_env_number_range("AGENT_PRE_CLICK_DELAY_RANGE", (0.05, 0.20)),
            post_action_settle_range=_env_number_range("AGENT_SETTLE_RANGE", (0.35, 0.80)),
            wait_action_range=_env_number_range("AGENT_WAIT_RANGE", (1.0, 2.0)),
        )


@dataclass(frozen=True)
class LoopSettings:
    """OODA loop budget and scroll behavior."""

    max_steps: int = 25
    max_runtime_seconds: float = 900.0
    scroll_clicks_default: int = 4
    scroll_clicks_max: int = 15

    @classmethod
    def from_env(cls) -> "LoopSettings":
        return cls(
            max_steps=_env_int("AGENT_MAX_STEPS", 25),
            max_runtime_seconds=_env_float("AGENT_MAX_RUNTIME_SECONDS", 900.0),
            scroll_clicks_default=_env_int("AGENT_SCROLL_DEFAULT", 4),
            scroll_clicks_max=_env_int("AGENT_SCROLL_MAX", 15),
        )


@dataclass(frozen=True)
class LoggingSettings:
    """Logging verbosity and optional file sink."""

    level: str = "INFO"
    log_to_file: bool = True
    file: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "LoggingSettings":
        raw_file = _env("AGENT_LOG_FILE")
        return cls(
            level=(_env("AGENT_LOG_LEVEL", "INFO") or "INFO").upper(),
            log_to_file=_env_bool("AGENT_LOG_TO_FILE", True),
            file=Path(raw_file) if raw_file else None,
        )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Immutable, fully validated configuration for one agent process."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    humanization: HumanizationSettings = field(default_factory=HumanizationSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, env_file: Optional[Path | str] = DEFAULT_ENV_FILE) -> "AppConfig":
        """Load ``.env`` (if present) and build a validated config."""
        if env_file is not None:
            path = Path(env_file)
            if path.is_file():
                load_dotenv(path, override=False)
                LOGGER.debug("Loaded environment from %s", path)
            elif path != DEFAULT_ENV_FILE:
                raise ConfigurationError(f"Env file not found: {path}")
        config = cls(
            llm=LLMSettings.from_env(),
            display=DisplaySettings.from_env(),
            safety=SafetySettings.from_env(),
            humanization=HumanizationSettings.from_env(),
            loop=LoopSettings.from_env(),
            logging=LoggingSettings.from_env(),
        )
        config.validate()
        return config

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Fail fast on impossible combinations before any OS access."""
        problems: list[str] = []

        if not 0.0 <= self.llm.temperature <= 2.0:
            problems.append("temperature must be within [0, 2]")
        if self.llm.max_output_tokens < 128:
            problems.append("max_output_tokens must be >= 128")
        if self.llm.api_max_attempts < 1:
            problems.append("api_max_attempts must be >= 1")
        if self.llm.schema_repair_attempts < 0:
            problems.append("schema_repair_attempts must be >= 0")
        if self.llm.base_url is not None and not self.llm.base_url.startswith(("http://", "https://")):
            problems.append("base_url must start with http:// or https://")

        if self.display.monitor_index < 1:
            problems.append("monitor_index must be >= 1 (0 means 'all monitors')")
        if self.display.max_image_side < 320:
            problems.append("max_image_side must be >= 320")
        if self.display.boundary_buffer < 0:
            problems.append("boundary_buffer must be >= 0")
        if self.display.display_scale is not None and self.display.display_scale <= 0:
            problems.append("display_scale must be > 0 when set")
        if self.display.screen_change_threshold < 0:
            problems.append("screen_change_threshold must be >= 0")
        if not 0 <= self.display.screen_change_fraction <= 1:
            problems.append("screen_change_fraction must be within [0, 1]")
        if not 0 <= self.display.screen_change_pixel_delta <= 255:
            problems.append("screen_change_pixel_delta must be within [0, 255]")
        if self.display.screen_change_rechecks < 0:
            problems.append("screen_change_rechecks must be >= 0")
        if self.display.screen_change_recheck_delay < 0:
            problems.append("screen_change_recheck_delay must be >= 0")

        if self.safety.max_identical_actions < 2:
            problems.append("max_identical_actions must be >= 2")
        if self.safety.max_state_repeats < 1:
            problems.append("max_state_repeats must be >= 1")
        if self.safety.max_futile_attempts < 1:
            problems.append("max_futile_attempts must be >= 1")
        if self.safety.max_consecutive_failures < 1:
            problems.append("max_consecutive_failures must be >= 1")
        if not self.safety.kill_switch_hotkey.strip():
            problems.append("kill_switch_hotkey must not be empty")

        if self.loop.max_steps < 1:
            problems.append("max_steps must be >= 1")
        if self.loop.max_runtime_seconds <= 0:
            problems.append("max_runtime_seconds must be > 0")
        if self.loop.scroll_clicks_max < 1:
            problems.append("scroll_clicks_max must be >= 1")
        if not 1 <= self.loop.scroll_clicks_default <= self.loop.scroll_clicks_max:
            problems.append("scroll_clicks_default must be within [1, scroll_clicks_max]")

        if self.logging.level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            problems.append(f"unknown log level {self.logging.level!r}")

        if problems:
            raise ConfigurationError("Invalid configuration: " + "; ".join(problems))

    # -- helpers ------------------------------------------------------------

    def require_api_key(self) -> str:
        """Return the API key, raising a clear error when it is absent."""
        if not self.llm.api_key:
            key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}.get(self.llm.provider)
            if key_var is None:
                raise ConfigurationError(f"Unsupported provider {self.llm.provider!r}.")
            raise ConfigurationError(
                f"No API key configured. Set {key_var} (or AGENT_API_KEY) in your .env file."
            )
        return self.llm.api_key

    def with_overrides(self, **sections: object) -> "AppConfig":
        """Return a copy with replaced sections, e.g. ``with_overrides(safety=...)``."""
        unknown = set(sections) - {"llm", "display", "safety", "humanization", "loop", "logging"}
        if unknown:
            raise ConfigurationError(f"Unknown config sections: {sorted(unknown)}")
        return replace(self, **sections)  # type: ignore[arg-type]

    def describe(self) -> dict[str, object]:
        """Serializable, secret-free summary suitable for logs and transcripts."""
        return {
            "provider": self.llm.provider,
            "model": self.llm.model,
            "base_url": self.llm.base_url,
            "temperature": self.llm.temperature,
            "hitl": self.safety.hitl,
            "kill_switch": (
                self.safety.kill_switch_hotkey if self.safety.kill_switch_enabled else "disabled"
            ),
            "failsafe": self.safety.failsafe,
            "roi": self.display.roi,
            "monitor_index": self.display.monitor_index,
            "display_scale": self.display.display_scale,
            "max_image_side": self.display.max_image_side,
            "boundary_buffer": self.display.boundary_buffer,
            "max_steps": self.loop.max_steps,
            "max_runtime_seconds": self.loop.max_runtime_seconds,
            "max_identical_actions": self.safety.max_identical_actions,
            "max_state_repeats": self.safety.max_state_repeats,
            "max_futile_attempts": self.safety.max_futile_attempts,
            "max_consecutive_failures": self.safety.max_consecutive_failures,
        }


# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "PIL",
    "urllib3",
    "keyboard",
)


def configure_logging(settings: LoggingSettings) -> None:
    """Install console (and optional file) logging exactly once."""
    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stdout)]
    if settings.log_to_file and settings.file is not None:
        settings.file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, settings.level, logging.INFO),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def attach_file_handler(path: Path) -> logging.Handler:
    """Add a run-scoped log file handler and return it (for later removal)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    LOGGER.info("Logging to %s", path)
    return handler
