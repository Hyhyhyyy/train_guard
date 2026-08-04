"""Dependency-free command-line parser definitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NoReturn

from . import __version__
from ._compat import LEGACY_COMMANDS
from .core.config import TEMPLATE_NAMES
from .core.exitcodes import EXIT_USAGE

STATUS_HELP = (
    "Results: PASS=checks passed (exit 0), WARN=review recommended (exit 1), "
    "FAIL=check failed (exit 2). Usage=3, configuration=4, runtime=5, refused overwrite=6."
)


class TrainGuardArgumentParser(argparse.ArgumentParser):
    """Argparse with the same actionable error contract as config validation."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(
            EXIT_USAGE,
            "FAIL\n"
            f"Problem: {message}\n"
            "Location: command line\n"
            "Fix: review the command help and correct the shown option\n",
        )


def _configured(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config (YAML also works when PyYAML is installed)",
    )


def _add(parser: argparse.ArgumentParser, *flags: str, **kwargs: Any) -> None:
    kwargs.setdefault("default", argparse.SUPPRESS)
    parser.add_argument(*flags, **kwargs)


def _data_check_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--data-root")
    _add(parser, "--annotation")
    _add(parser, "--sample-limit", type=int)
    _add(parser, "--full-scan", action=argparse.BooleanOptionalAction)
    _add(parser, "--compute-hash", action=argparse.BooleanOptionalAction)
    _add(parser, "--verify-images", action=argparse.BooleanOptionalAction)
    _add(parser, "--max-image-pixels", type=int)
    _add(parser, "--max-media-files", type=int)
    _add(parser, "--max-scan-bytes", type=int)
    _add(parser, "--allow-external-media", action=argparse.BooleanOptionalAction)
    _add(parser, "--group-id-field", dest="group_id")
    _add(parser, "--split-field", dest="split")
    _add(parser, "--media-field", dest="media")
    _add(parser, "--messages-field", dest="messages")
    _add(parser, "--report-dir")
    _add(parser, "--cache-db")
    _add(parser, "--issues-jsonl")


def _run_watch_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--once", action=argparse.BooleanOptionalAction)
    _add(parser, "--interval", type=int)
    _add(parser, "--pid", type=int)
    _add(parser, "--log-file")
    _add(parser, "--framework", choices=("generic", "huggingface", "transformers", "llamafactory"))
    _add(parser, "--output-dir")
    _add(parser, "--expected-gpus", type=int)
    _add(parser, "--stale-log-minutes", type=float)
    _add(parser, "--disk-free-gb-threshold", type=float)
    _add(parser, "--run-id")
    _add(parser, "--state-db")
    _add(parser, "--webhook-url")
    _add(parser, "--prometheus-file")
    _add(parser, "--otel-file")
    _add(parser, "--reliability", action=argparse.BooleanOptionalAction)
    _add(parser, "--notification-every", type=int)
    _add(parser, "--step-stall-seconds", type=float)
    _add(parser, "--gpu-overheat-celsius", type=float)
    _add(parser, "--checkpoint-stale-seconds", type=float)


def _run_check_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--output-dir")
    _add(parser, "--framework", choices=("generic", "huggingface", "transformers", "llamafactory"))
    _add(parser, "--expected-steps", type=int)
    _add(parser, "--training-type", choices=("auto", "peft", "lora", "qlora", "full"))
    _add(parser, "--json-output")
    _add(parser, "--html-output")


def _run_compare_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--left")
    _add(parser, "--right")
    _add(parser, "--framework", choices=("generic", "huggingface", "transformers", "llamafactory"))
    _add(parser, "--json-output")


def _eval_args(parser: argparse.ArgumentParser) -> None:
    _configured(parser)
    _add(parser, "--predictions")
    _add(parser, "--references")
    _add(parser, "--prediction-field")
    _add(parser, "--reference-field")
    _add(parser, "--group-id-field", dest="group_id")
    _add(parser, "--label-field")
    _add(parser, "--predicted-label-field")
    _add(parser, "--keywords", help="Comma-separated keywords")
    _add(parser, "--report-dir")
    _add(parser, "--sample-limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    """Build the dependency-free argparse parser."""
    parser = TrainGuardArgumentParser(
        prog="train-guard",
        description="Local-first LLM/VLM training reliability and guarded recovery.",
        epilog=STATUS_HELP,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser(
        "init", help="Generate a validated starter configuration", epilog=STATUS_HELP
    )
    command.add_argument("--output", type=Path, default=Path("train-guard.json"))
    command.add_argument("--template", choices=TEMPLATE_NAMES, default="generic")
    command.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    command.set_defaults(_handler="init")

    command = sub.add_parser(
        "doctor", help="Environment and model integrity check", epilog=STATUS_HELP
    )
    _configured(command)
    _add(command, "--model-path")
    _add(command, "--expected-gpus", type=int)
    _add(command, "--json-output")
    command.set_defaults(_handler="doctor")

    data = sub.add_parser("data", help="Dataset commands")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    command = data_sub.add_parser(
        "check", help="Dataset/media/message integrity check", epilog=STATUS_HELP
    )
    _data_check_args(command)
    command.set_defaults(_handler="data_check")
    command = data_sub.add_parser(
        "inventory", help="Streaming dataset inventory", epilog=STATUS_HELP
    )
    _configured(command)
    _add(command, "--annotation")
    _add(command, "--sample-limit", type=int)
    _add(command, "--group-id-field", dest="group_id")
    _add(command, "--report-dir")
    command.set_defaults(_handler="data_inventory")
    command = data_sub.add_parser(
        "compare", help="Compare two annotation files", epilog=STATUS_HELP
    )
    _configured(command)
    _add(command, "--left")
    _add(command, "--right")
    _add(command, "--sample-limit", type=int)
    _add(command, "--group-id-field", dest="group_id")
    _add(command, "--report-dir")
    command.set_defaults(_handler="data_compare")

    run = sub.add_parser("run", help="Training run commands")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    command = run_sub.add_parser("watch", help="Read-only training watch", epilog=STATUS_HELP)
    _run_watch_args(command)
    command.set_defaults(_handler="run_watch")
    command = run_sub.add_parser(
        "snapshot", help="Collect one reliability snapshot", epilog=STATUS_HELP
    )
    _run_watch_args(command)
    command.set_defaults(_handler="run_snapshot")
    command = run_sub.add_parser(
        "check", help="Check whether a training run completed", epilog=STATUS_HELP
    )
    _run_check_args(command)
    command.set_defaults(_handler="run_check")
    command = run_sub.add_parser(
        "compare", help="Compare two run output directories", epilog=STATUS_HELP
    )
    _run_compare_args(command)
    command.set_defaults(_handler="run_compare")
    command = run_sub.add_parser(
        "status",
        help="Show the persistent status of a training run",
        epilog=STATUS_HELP,
    )
    command.add_argument("--state-db", type=Path, required=True)
    command.add_argument("--run-id")
    command.add_argument("--json-output", type=Path)
    command.set_defaults(_handler="run_status")
    command = run_sub.add_parser(
        "launch",
        help="Run preflight, monitored training, recovery, and acceptance as one workflow",
        epilog=STATUS_HELP,
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--framework",
        choices=("generic", "huggingface", "transformers", "llamafactory"),
        default="generic",
    )
    command.add_argument(
        "--training-type", choices=("auto", "peft", "lora", "qlora", "full"), default="auto"
    )
    command.add_argument("--expected-steps", type=int)
    command.add_argument("--expected-gpus", type=int)
    command.add_argument("--model-path", type=Path)
    command.add_argument("--run-id")
    command.add_argument("--state-db", type=Path)
    command.add_argument("--summary-out", type=Path)
    command.add_argument("--audit-log", type=Path)
    command.add_argument("--log-file", type=Path)
    command.add_argument("--monitor-interval", type=int, default=5)
    command.add_argument("--strict-preflight", action="store_true")
    command.add_argument("--restart", action="store_true")
    command.add_argument("--max-restarts", type=int, default=0)
    command.add_argument("--restart-window-seconds", type=float, default=3600.0)
    command.add_argument("--checkpoint-dir", type=Path)
    command.add_argument("--health-file", type=Path)
    command.add_argument("--health-max-age", type=float, default=30.0)
    command.add_argument("--health-timeout", type=float, default=120.0)
    command.add_argument("--health-interval", type=float, default=2.0)
    command.add_argument(
        "--required-checkpoint-file",
        action="append",
        default=[],
        help="Checkpoint-relative file required before restart; repeatable",
    )
    command.add_argument("--enable-control", action="store_true")
    command.add_argument("--seed")
    command.add_argument("training_command", nargs=argparse.REMAINDER)
    command.set_defaults(_handler="run_launch")

    command = run_sub.add_parser(
        "supervise",
        help="Launch a training command with bounded, checkpoint-gated restart",
        epilog=STATUS_HELP,
    )
    command.add_argument("--restart", action="store_true", help="Explicitly enable restart")
    command.add_argument("--max-restarts", type=int, default=0)
    command.add_argument("--restart-window-seconds", type=float, default=3600.0)
    command.add_argument("--run-id")
    command.add_argument(
        "--enable-control",
        action="store_true",
        help="Consume allowlisted dashboard/TUI control requests for this run",
    )
    command.add_argument(
        "--state-db",
        type=Path,
        help="Persistent restart budget database (defaults beside audit log)",
    )
    command.add_argument("--checkpoint-dir", type=Path)
    command.add_argument(
        "--health-file",
        type=Path,
        help="Heartbeat file that must become fresh after a restart",
    )
    command.add_argument("--health-max-age", type=float, default=30.0)
    command.add_argument("--health-timeout", type=float, default=120.0)
    command.add_argument("--health-interval", type=float, default=2.0)
    command.add_argument(
        "--required-checkpoint-file",
        action="append",
        default=[],
        help="Checkpoint-relative file required before restart; repeatable",
    )
    command.add_argument(
        "--audit-log",
        type=Path,
        default=Path("train_guard_supervisor.jsonl"),
    )
    command.add_argument("training_command", nargs=argparse.REMAINDER)
    command.set_defaults(_handler="run_supervise")

    command = sub.add_parser("eval", help="Evaluate predictions vs references", epilog=STATUS_HELP)
    _eval_args(command)
    command.set_defaults(_handler="eval")

    command = sub.add_parser(
        "manifest", help="Write run manifest and experiment fingerprint", epilog=STATUS_HELP
    )
    _configured(command)
    _add(command, "--output-dir")
    _add(command, "--framework", choices=("generic", "huggingface", "transformers", "llamafactory"))
    _add(command, "--manifest-out")
    _add(command, "--expected-steps", type=int)
    _add(command, "--seed")
    command.set_defaults(_handler="manifest")

    command = sub.add_parser("bundle-info", help="Show deploy/version info")
    command.set_defaults(_handler="bundle_info")

    command = sub.add_parser("show", help="Serve the localhost-only reliability dashboard")
    command.add_argument("--state-db", type=Path, required=True)
    command.add_argument("--host", choices=("127.0.0.1", "::1", "localhost"), default="127.0.0.1")
    command.add_argument("--port", type=int, default=8765)
    command.add_argument(
        "--enable-control",
        action="store_true",
        help="Enable authenticated control for supervised runs",
    )
    command.set_defaults(_handler="show")

    command = sub.add_parser("tui", help="Run the optional terminal monitoring dashboard")
    command.add_argument("--state-db", type=Path, required=True)
    command.add_argument("--run-id")
    command.add_argument("--enable-control", action="store_true")
    command.set_defaults(_handler="tui")

    command = sub.add_parser("compare", help="Alias for 'run compare'", epilog=STATUS_HELP)
    _run_compare_args(command)
    command.set_defaults(_handler="run_compare")

    alias_arguments = (
        ("precheck", _data_check_args),
        ("monitor", _run_watch_args),
        ("postcheck", _run_check_args),
        ("evaluate", _eval_args),
    )
    for name, add_args in alias_arguments:
        legacy = LEGACY_COMMANDS[name]
        command = sub.add_parser(
            name,
            help=f"Deprecated alias for '{legacy.replacement}'",
            epilog=STATUS_HELP,
        )
        add_args(command)
        command.set_defaults(
            _handler=legacy.canonical_handler,
            _legacy_command=name,
        )
    return parser


HANDLER_SECTIONS = {
    "doctor": ("doctor",),
    "data_check": ("data", "check"),
    "data_inventory": ("data", "inventory"),
    "data_compare": ("data", "compare"),
    "run_watch": ("run", "watch"),
    "run_snapshot": ("run", "watch"),
    "run_check": ("run", "check"),
    "run_compare": ("run", "compare"),
    "eval": ("eval",),
    "manifest": ("manifest",),
}


__all__ = [
    "HANDLER_SECTIONS",
    "STATUS_HELP",
    "TrainGuardArgumentParser",
    "build_parser",
]
