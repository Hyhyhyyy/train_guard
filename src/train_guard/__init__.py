"""Local-first LLM/VLM training reliability and guarded recovery toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.6.0rc1"
__min_python__ = (3, 10)

if TYPE_CHECKING:
    from .api import (
        ReliabilitySession,
        check_dataset,
        check_run,
        evaluate_predictions,
        watch_snapshot,
    )

_API_EXPORTS = frozenset(
    {
        "ReliabilitySession",
        "check_dataset",
        "check_run",
        "evaluate_predictions",
        "watch_snapshot",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _API_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".api", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_API_EXPORTS})


__all__ = [
    "ReliabilitySession",
    "__min_python__",
    "__version__",
    "check_dataset",
    "check_run",
    "evaluate_predictions",
    "watch_snapshot",
]
