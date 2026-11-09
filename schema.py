"""Action space, VLM output validation, and shared exception types.

The VLM is *never* trusted: every response is parsed as JSON and validated by
Pydantic before it can reach an OS driver. Invalid shapes are rejected with a
precise message so the agent can ask the model to repair its own output.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Final, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Point = Tuple[int, int]
ActionSignature = Tuple[str, Optional[Tuple[Point, ...]], Optional[str]]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentError(RuntimeError):
    """Base class for every recoverable agent failure."""


class ActionValidationError(AgentError):
    """The model returned text that does not satisfy the action schema."""


class ActionExecutionError(AgentError):
    """An OS-level driver call failed while executing a validated action."""


class StagnationError(AgentError):
    """The agent repeated the same action too many times without progress."""


class SafetyAbort(AgentError):
    """A safety mechanism (fail-safe, kill-switch, boundary policy) stopped the run."""


class CaptureError(AgentError):
    """Screen capture failed (locked session, invalid ROI, driver error)."""


class ConfigurationError(AgentError):
    """The runtime environment or configuration cannot support a run."""


#: Models report click points on a normalized 0-1000 grid (the convention vision
#: models such as Gemini are trained on), never in screenshot pixels.
COORDINATE_SPACE: Final[int] = 1000

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """The complete action space exposed to the VLM."""

    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    TYPE = "TYPE"
    PRESS_KEY = "PRESS_KEY"
    SCROLL = "SCROLL"
    DRAG = "DRAG"
    WAIT = "WAIT"
    DONE = "DONE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


#: Actions that must carry exactly one screen point.
_SINGLE_POINT_ACTIONS: Final[frozenset[ActionType]] = frozenset(
    {ActionType.CLICK, ActionType.DOUBLE_CLICK}
)
#: Actions that must not carry coordinates.
_POINTLESS_ACTIONS: Final[frozenset[ActionType]] = frozenset(
    {ActionType.PRESS_KEY, ActionType.WAIT, ActionType.DONE, ActionType.ESCALATE_TO_HUMAN}
)


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------


class AgentAction(BaseModel):
    """One validated step emitted by the vision-language model."""

    model_config = ConfigDict(extra="ignore")

    thought_process: str = Field(
        ...,
        min_length=1,
        max_length=2_000,
        description="Short rationale grounded in the screenshot.",
    )
    action_type: ActionType = Field(..., description="One of the nine allowed actions.")
    coordinates: Optional[list[Point]] = Field(
        default=None,
        description=(
            "Points on a normalized 0-1000 grid, where x=0/1000 is the left/right "
            "edge and y=0/1000 is the top/bottom edge of the screenshot. Both values "
            "are integers in [0, 1000]. One point for CLICK/DOUBLE_CLICK/TYPE, two "
            "points (drag start, drag end) for DRAG. Null for WAIT."
        ),
    )
    text_input: Optional[str] = Field(
        default=None,
        max_length=8_000,
        description="Text to type. Required for TYPE; optional free-text note for ESCALATE.",
    )
    key_input: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Key or chord to press. Required for PRESS_KEY, e.g. \"win\", \"enter\", "
            "\"esc\", \"tab\", \"ctrl+s\", \"alt+tab\". Join chord members with '+'."
        ),
    )
    scroll_amount: Optional[int] = Field(
        default=None,
        ge=-15,
        le=15,
        description=(
            "Wheel clicks for SCROLL: positive scrolls content up (wheel away from user), "
            "negative scrolls down. Ignored for other actions."
        ),
    )

    # -- coercion -----------------------------------------------------------

    @field_validator("thought_process")
    @classmethod
    def _thought_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("thought_process must contain a short explanation")
        return cleaned

    @field_validator("text_input", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return str(value)
        if not isinstance(value, str):
            raise ValueError("text_input must be a string or null")
        return value

    @field_validator("key_input")
    @classmethod
    def _normalize_key(cls, value: Optional[str]) -> Optional[str]:
        """Accept ``Ctrl + S`` / ``CTRL+S`` and normalize to ``ctrl+s``."""
        if value is None:
            return None
        cleaned = value.strip().lower().replace(" ", "")
        if not cleaned:
            raise ValueError("key_input must name at least one key")
        tokens = cleaned.split("+")
        if len(tokens) > 4:
            raise ValueError("key_input accepts at most 4 keys in one chord")
        for token in tokens:
            if not token or not re.fullmatch(r"[a-z0-9_-]{1,20}", token):
                raise ValueError(
                    f"key_input token {token!r} is not a plain key name "
                    "(use names like win, enter, esc, tab, f5, ctrl+s)"
                )
        return "+".join(tokens)

    @field_validator("coordinates", mode="before")
    @classmethod
    def _coerce_coordinates(cls, value: Any) -> Any:
        """Accept the many shapes VLMs produce and normalize to ``[[x, y], ...]``.

        Tolerated inputs include ``null``, ``[]``, ``[x, y]``, ``[[x, y]]``,
        ``[{"x": 1, "y": 2}]``, and JSON-encoded strings of any of those.
        """
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"coordinates string is not valid JSON: {exc.msg}") from exc
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("coordinates must be a list of [x, y] pairs or null")
        if len(value) == 0:
            return None
        # A flat pair: [x, y] -> [[x, y]]
        if len(value) == 2 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            value = [value]

        points: list[list[int]] = []
        for item in value:
            if isinstance(item, dict):
                if "x" not in item or "y" not in item:
                    raise ValueError("coordinate objects must have 'x' and 'y' keys")
                item = [item["x"], item["y"]]
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("each coordinate must be an [x, y] pair")
            x, y = item
            if isinstance(x, bool) or isinstance(y, bool):
                raise ValueError("coordinates must be numbers, not booleans")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError("coordinates must be numeric")
            points.append([int(round(float(x))), int(round(float(y)))])
        for x, y in points:
            if not (0 <= x <= COORDINATE_SPACE and 0 <= y <= COORDINATE_SPACE):
                raise ValueError(
                    "coordinates must lie on the normalized 0-1000 grid "
                    f"(got [{x}, {y}]); values outside that range are not pixels"
                )
        return points

    # -- cross-field rules --------------------------------------------------

    @model_validator(mode="after")
    def _enforce_action_contract(self) -> "AgentAction":
        action = self.action_type
        points = self.coordinates

        if action in _SINGLE_POINT_ACTIONS:
            if points is None or len(points) != 1:
                raise ValueError(f"{action.value} requires exactly one [x, y] coordinate")
        elif action is ActionType.DRAG:
            if points is None or len(points) != 2:
                raise ValueError("DRAG requires exactly two coordinates: [start, end]")
        elif action is ActionType.TYPE:
            if not self.text_input or not self.text_input.strip():
                raise ValueError("TYPE requires a non-empty text_input")
            if points is not None and len(points) > 1:
                raise ValueError("TYPE accepts at most one click target coordinate")
        elif action is ActionType.PRESS_KEY:
            if not self.key_input or not self.key_input.strip():
                raise ValueError(
                    "PRESS_KEY requires a non-empty key_input such as \"win\", "
                    "\"enter\" or \"ctrl+s\""
                )
            if points:
                raise ValueError("PRESS_KEY must not include coordinates")
        elif action is ActionType.SCROLL:
            if points is not None and len(points) > 1:
                raise ValueError("SCROLL accepts at most one cursor coordinate")
            if self.scroll_amount is None:
                self.scroll_amount = 3
            if self.scroll_amount == 0:
                raise ValueError("SCROLL requires a non-zero scroll_amount")
        else:  # WAIT / DONE / ESCALATE_TO_HUMAN
            if points:
                raise ValueError(f"{action.value} must not include coordinates")

        if action is not ActionType.SCROLL:
            self.scroll_amount = None
        if action is not ActionType.PRESS_KEY:
            self.key_input = None
        return self

    # -- helpers ------------------------------------------------------------

    def signature(self) -> ActionSignature:
        """Stable identity used for stagnation detection.

        Coordinates are compared exactly, as required by the safety policy; text
        only matters for TYPE so that e.g. two identical clicks are recognized.
        """
        points: Optional[Tuple[Point, ...]] = (
            tuple(self.coordinates) if self.coordinates else None
        )
        if self.action_type is ActionType.TYPE:
            text = self.text_input
        elif self.action_type is ActionType.PRESS_KEY:
            text = self.key_input
        else:
            text = None
        return (self.action_type.value, points, text)

    def describe(self) -> str:
        """Single-line human readable summary for console and logs."""
        parts: list[str] = [self.action_type.value]
        if self.coordinates:
            parts.append("at " + ", ".join(f"({x}, {y})" for x, y in self.coordinates))
        if self.scroll_amount is not None:
            parts.append(f"amount={self.scroll_amount:+d}")
        if self.key_input is not None and self.action_type is ActionType.PRESS_KEY:
            parts.append(f'key="{self.key_input}"')
        elif self.text_input is not None and self.action_type is ActionType.TYPE:
            preview = self.text_input.replace("\n", "\\n")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            parts.append(f'text="{preview}" ({len(self.text_input)} chars)')
        elif self.text_input:
            parts.append(f'note="{self.text_input[:80]}"')
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FENCED_JSON: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE
)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a model response that may contain prose/fences.

    Raises:
        ActionValidationError: if no parseable JSON object is present.
    """
    if raw is None or not raw.strip():
        raise ActionValidationError("model returned an empty response")

    match = _FENCED_JSON.search(raw)
    candidate = match.group("body") if match else raw
    candidate = candidate.strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ActionValidationError("model response contains no JSON object")

    fragment = candidate[start : end + 1]
    try:
        payload = json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise ActionValidationError(
            f"model response is not valid JSON ({exc.msg} at char {exc.pos})"
        ) from exc

    if not isinstance(payload, dict):
        raise ActionValidationError("model response must be a single JSON object")

    # Tolerate a nested wrapper such as {"action": {...}}.
    if "action_type" not in payload:
        nested = payload.get("action")
        if isinstance(nested, dict):
            payload = nested
    return payload


def _format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "response"
        messages.append(f"{location}: {detail['msg']}")
    return "; ".join(messages)


def parse_action(raw: str) -> AgentAction:
    """Parse and validate raw VLM output into an :class:`AgentAction`."""
    payload = extract_json_object(raw)
    try:
        return AgentAction.model_validate(payload)
    except ValidationError as exc:
        raise ActionValidationError(_format_validation_error(exc)) from exc


def action_schema_for_prompt() -> str:
    """Compact JSON-schema description embedded in the system prompt."""
    return json.dumps(AgentAction.model_json_schema(), separators=(",", ":"), ensure_ascii=False)


def build_repair_message(error: Exception, previous: str) -> str:
    """Instruction sent back to the model after an invalid response."""
    return (
        "Your previous answer was rejected by the JSON schema validator.\n"
        f"Validator error: {error}\n"
        "Respond again with ONE corrected JSON object only. No prose, no markdown "
        "fences, no extra keys. Coordinates must be integers on the normalized "
        "0-1000 grid of the screenshot you were shown, never raw pixels."
    )
