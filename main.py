"""Command-line entry point for the AutoClick desktop agent.

Usage:
    python main.py "Open Notepad and type a haiku"
    python main.py --diagnose
    python main.py "Summarize the open document" --hitl --roi 100,100,1200,800

Exit codes:
    0   goal completed
    1   configuration/runtime error
    2   escalated to human / stagnated / step budget exhausted
    130 aborted by the operator (signal, fail-safe, kill-switch)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from agent import AgentResult, DesktopAgent, RunStatus
from config import AppConfig, configure_logging, parse_roi
from schema import AgentError, ConfigurationError
from vision_llm import ScreenCapture, describe_environment

LOGGER = logging.getLogger(__name__)

EXIT_CODES: dict[RunStatus, int] = {
    RunStatus.DONE: 0,
    RunStatus.ERROR: 1,
    RunStatus.ESCALATED: 2,
    RunStatus.STAGNATED: 2,
    RunStatus.MAX_STEPS: 2,
    RunStatus.ABORTED: 130,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoclick",
        description="Goal-driven AI desktop agent (vision-language model + PyAutoGUI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("goal", nargs="?", help="Natural-language goal for the agent.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print monitor, scaling, and ROI diagnostics, then exit.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(Path(__file__).resolve().parent / ".env"),
        help="Path to a .env file with API keys and settings.",
    )
    parser.add_argument(
        "--hitl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require y/n confirmation before every action.",
    )
    parser.add_argument(
        "--roi",
        metavar="LEFT,TOP,WIDTH,HEIGHT",
        help="Capture only this region (physical pixels, MSS virtual-desktop coordinates).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override the physical-to-logical display multiplier (e.g. 1.5, 2.0).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="MSS monitor index to capture (1 = primary).",
    )
    parser.add_argument("--model", type=str, default=None, help="Override the VLM model name.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override the step budget.")
    parser.add_argument(
        "--max-runtime", type=float, default=None, help="Override the wall-clock budget (seconds)."
    )
    parser.add_argument(
        "--screenshots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save per-step screenshots and a transcript under runs/.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Console log verbosity.",
    )
    return parser


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Return a copy of ``config`` with explicit CLI flags applied."""
    display = config.display
    safety = config.safety
    loop = config.loop
    llm = config.llm
    logging_settings = config.logging

    if args.roi:
        display = replace(display, roi=parse_roi(args.roi))
    if args.scale is not None:
        display = replace(display, display_scale=args.scale)
    if args.monitor is not None:
        display = replace(display, monitor_index=args.monitor)
    if args.screenshots is not None:
        display = replace(display, save_screenshots=args.screenshots)
    if args.hitl is not None:
        safety = replace(safety, hitl=args.hitl)
    if args.max_steps is not None:
        loop = replace(loop, max_steps=args.max_steps)
    if args.max_runtime is not None:
        loop = replace(loop, max_runtime_seconds=args.max_runtime)
    if args.model:
        llm = replace(llm, model=args.model)
    if args.log_level:
        logging_settings = replace(logging_settings, level=args.log_level)

    return replace(
        config,
        llm=llm,
        display=display,
        safety=safety,
        loop=loop,
        logging=logging_settings,
    )


def print_result(result: AgentResult) -> None:
    """Summarize a finished run on stdout (logs already hold the details)."""
    print("\n" + "=" * 70)
    print(f"STATUS : {result.status.value}")
    print(f"STEPS  : {result.steps}")
    print(f"TIME   : {result.elapsed_seconds:.1f}s")
    print(f"RESULT : {result.message}")
    if result.run_dir is not None:
        print(f"ARTIFACTS: {result.run_dir}")
    print("=" * 70)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.diagnose and not args.goal:
        parser.error("a GOAL is required unless --diagnose is used")

    try:
        config = AppConfig.load(args.env_file)
        config = apply_overrides(config, args)
        config.validate()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(config.logging)

    if args.diagnose:
        capture: Optional[ScreenCapture] = None
        try:
            capture = ScreenCapture(config.display)
            print(describe_environment(capture))
            return 0
        except AgentError as exc:
            LOGGER.error("Diagnostics failed: %s", exc)
            return 1
        except Exception as exc:  # pragma: no cover - driver specific
            LOGGER.error("Diagnostics failed: %s", exc, exc_info=True)
            return 1
        finally:
            if capture is not None:
                capture.close()

    LOGGER.debug("Effective configuration: %s", config.describe())

    try:
        agent = DesktopAgent(config)
    except (ConfigurationError, AgentError) as exc:
        LOGGER.error("Could not start the agent: %s", exc)
        return 1

    try:
        result = agent.run(args.goal or "")
    except ConfigurationError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.critical("Interrupted by the operator.")
        return 130
    except AgentError as exc:
        LOGGER.error("Agent error: %s", exc)
        return 1

    print_result(result)
    return EXIT_CODES.get(result.status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
