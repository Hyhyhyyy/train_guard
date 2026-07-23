#!/usr/bin/env python3
"""Development-tree launcher; publish only the validated release candidate."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from train_guard.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
