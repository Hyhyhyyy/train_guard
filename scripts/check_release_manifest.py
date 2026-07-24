#!/usr/bin/env python3
"""Generate or validate the fixed candidate-release allowlist and SHA256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "release"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "release_manifest.json"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[dict[str, str], list[str]]:
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
            if (
                parts & FORBIDDEN_PARTS
                or suffix in FORBIDDEN_SUFFIXES
                or bad_sqlite
                or path.is_symlink()
            ):
                rejected.append(rel)
            elif path.is_file():
                files.append(rel)
    return {rel: sha256_file(root / rel) for rel in files}, rejected


def validate_manifest(root: Path) -> tuple[list[str], list[str]]:
    """Compatibility helper returning sorted paths and policy rejections."""
    files, rejected = inventory(root)
    return list(files), rejected


def load_manifest(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is missing or invalid JSON") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 1
        or raw.get("algorithm") != "sha256"
    ):
        raise ValueError("manifest requires schema_version 1 and algorithm sha256")
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest files must be a non-empty object")
    normalized: dict[str, str] = {}
    for rel, digest in files.items():
        if (
            not isinstance(rel, str)
            or Path(rel).is_absolute()
            or ".." in Path(rel).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("manifest contains an unsafe path or invalid SHA256")
        normalized[rel.replace("\\", "/")] = digest
    return dict(sorted(normalized.items()))


def write_manifest(path: Path, files: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "algorithm": "sha256", "files": dict(sorted(files.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate candidate release contents")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Replace the manifest from the policy-clean candidate tree",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print("release manifest error: candidate directory does not exist", file=sys.stderr)
        return 2
    files, rejected = inventory(root)
    if rejected:
        for rel in rejected:
            print(f"REJECTED {rel}")
        return 1
    if args.generate:
        write_manifest(args.manifest, files)
        print(f"wrote {args.manifest}")
        return 0
    try:
        expected = load_manifest(args.manifest)
    except ValueError as exc:
        print(f"release manifest error: {exc}", file=sys.stderr)
        return 2
    missing = sorted(set(expected) - set(files))
    unexpected = sorted(set(files) - set(expected))
    changed = sorted(rel for rel in set(files) & set(expected) if files[rel] != expected[rel])
    for rel in missing:
        print(f"MISSING {rel}")
    for rel in unexpected:
        print(f"UNEXPECTED {rel}")
    for rel in changed:
        print(f"SHA256_MISMATCH {rel}")
    if missing or unexpected or changed:
        return 1
    for rel, digest in files.items():
        print(f"{digest}  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
