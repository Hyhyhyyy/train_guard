"""One-release compatibility boundary for deprecated CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LegacyCommand:
    canonical_handler: str
    replacement: str


LEGACY_COMMANDS: Dict[str, LegacyCommand] = {
    "precheck": LegacyCommand("data_check", "data check"),
    "monitor": LegacyCommand("run_watch", "run watch"),
    "postcheck": LegacyCommand("run_check", "run check"),
    "evaluate": LegacyCommand("eval", "eval"),
}
LEGACY_HANDLERS = {
    f"legacy_{name}": (name, command.canonical_handler) for name, command in LEGACY_COMMANDS.items()
}

COMPATIBILITY_REMOVAL_VERSION = "0.6.0"


def deprecation_message(name: str) -> str:
    command = LEGACY_COMMANDS[name]
    return (
        f"[DEPRECATED] {name} ->! {command.replacement}; "
        f"scheduled for removal in {COMPATIBILITY_REMOVAL_VERSION}; "
        "see docs/MIGRATION.md"
    )


__all__ = [
    "COMPATIBILITY_REMOVAL_VERSION",
    "LEGACY_COMMANDS",
    "LEGACY_HANDLERS",
    "LegacyCommand",
    "deprecation_message",
]
