"""Optional training-framework integrations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .huggingface import TrainGuardCallback


def __getattr__(name: str) -> Any:
    if name != "TrainGuardCallback":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(".huggingface", __name__).TrainGuardCallback
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), "TrainGuardCallback"})


__all__ = ["TrainGuardCallback"]
