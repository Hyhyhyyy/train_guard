"""Compatibility facade for split training-run command modules."""

from .check import run_run_check
from .manifest import run_manifest, run_run_compare
from .watch import (
    _handle_signal,
    _reliability_values,
    cmd_run_watch,
    collect_watch_sample,
    query_nvidia_smi,
)

__all__ = [
    "cmd_run_watch",
    "collect_watch_sample",
    "query_nvidia_smi",
    "run_manifest",
    "run_run_check",
    "run_run_compare",
]
