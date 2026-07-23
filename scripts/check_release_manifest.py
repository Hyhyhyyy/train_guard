#!/usr/bin/env python3
"""Validate and print the complete candidate-release file manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "release"
FORBIDDEN_PARTS = {
    "private",
    "reports",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    ".venv",
    "venv",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".sqlite", ".sqlite3", ".log"}


def validate_manifest(root: Path) -> tuple[list[str], list[str]]:
    files: list[str] = []
    rejected: list[str] = []
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirs.sort()
        names.sort()
        for name in dirs + names:
            path = current_path / name
            rel = path.relative_to(root).as_posix()
            parts = {part.lower() for part in Path(rel).parts}
            suffix = path.suffix.lower()
            bad_sqlite = name.lower().startswith(".coverage") or ".sqlite" in name.lower()
            if parts & FORBIDDEN_PARTS or suffix in FORBIDDEN_SUFFIXES or bad_sqlite or path.is_symlink():
                rejected.append(rel)
            elif path.is_file():
                files.append(rel)
    return files, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate candidate release contents")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_RELEASE_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print("release manifest error: candidate directory does not exist", file=sys.stderr)
        return 2
    files, rejected = validate_manifest(root)
    if rejected:
        for rel in rejected:
            print(f"REJECTED {rel}")
        return 1
    for rel in files:
        print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
