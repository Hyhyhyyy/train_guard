#!/usr/bin/env python3
"""Privacy gate for a complete candidate release tree.

Findings deliberately contain metadata only. Matched text is never printed.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "release"
MAX_FILE_SIZE = 8 * 1024 * 1024
SOURCE_ENTRIES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "src",
    "tests",
    "scripts",
    "configs",
    "docs",
    "examples",
)
FORBIDDEN_SOURCE_NAMES = {"legacy", "private", "reports"}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "venv",
}


@dataclass(frozen=True)
class Rule:
    number: str
    category: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    category: str


RULES = (
    Rule(
        "R001",
        "credential",
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|authorization)"
            r"\s*[:=]\s*['\"]?"
            r"(?!(?:str|bytes|bool|int|float|object|Any|Optional|Union|Literal|Annotated)\b)"
            r"(?!<redacted>|example|placeholder|dummy|test)[^\s'\"]{4,}"
        ),
    ),
    Rule(
        "R002",
        "absolute-path",
        re.compile(
            r"(?<![\w./:])(?:[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s'\"]+"
            r"|/(?:home/[A-Za-z0-9._-]+|Users/[A-Za-z0-9._-]+|root|mnt/[A-Za-z0-9._-]+)"
            r"(?:/[^\s'\"]*)?)"
        ),
    ),
    Rule(
        "R003",
        "domain-specific",
        re.compile(
            r"(?i)\b(?:patient(?:_id)?|medical(?:_keywords)?|diagnosis|liver|hepatic|tumou?r|lesion)\b"
            r"|患者|病人|医疗|医学|诊断|肝脏|肝病|肿瘤|病理"
        ),
    ),
    Rule(
        "R004",
        "personal-data",
        re.compile(r"(?i)\b(?:full[_ -]?name|date[_ -]?of[_ -]?birth|social[_ -]?security)\b|姓名|出生日期|身份证"),
    ),
    Rule(
        "R008",
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    Rule(
        "R009",
        "server-address",
        re.compile(
            r"(?i)\b(?:ssh|sftp)://[^\s'\"]+"
            r"|(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?![\d.])"
            r"|\b(?:server|endpoint|base_url)\s*[:=]\s*['\"]"
            r"(?!localhost\b|127\.0\.0\.1\b|example\b|<redacted>\b)[^'\"]{3,}['\"]"
        ),
    ),
)


def _is_python_joined_name_identifier(path: Path, line: str, start: int, end: int) -> bool:
    """Return whether a joined personal-name match is a Python NAME token."""
    joined_name = "full" + "name"
    if path.suffix.lower() != ".py" or line[start:end].casefold() != joined_name:
        return False
    try:
        tokens = tokenize.generate_tokens(io.StringIO(line).readline)
        return any(
            token.type == tokenize.NAME
            and token.string.casefold() == joined_name
            and token.start[1] == start
            and token.end[1] == end
            for token in tokens
        )
    except (IndentationError, tokenize.TokenError):
        return False


def _matches_rule(rule: Rule, path: Path, line: str) -> bool:
    for match in rule.pattern.finditer(line):
        stripped = line.lstrip()
        if path.resolve() == Path(__file__).resolve() and (
            stripped.startswith('r"') or "re.compile(r\"" in stripped
        ):
            continue
        if rule.number == "R004" and _is_python_joined_name_identifier(
            path, line, match.start(), match.end()
        ):
            continue
        return True
    return False


class AllowlistError(ValueError):
    """Raised when an allowlist is ambiguous or lacks a reason."""


def load_allowlist(path: Path | None) -> set[tuple[str, str]]:
    """Load exact path/rule exemptions; every entry must explain why."""
    if path is None:
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AllowlistError(f"invalid allowlist: {type(exc).__name__}") from exc
    if not isinstance(raw, list):
        raise AllowlistError("allowlist must be a JSON array")
    result: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise AllowlistError("allowlist entries must be objects")
        rel = item.get("path")
        rule = item.get("rule")
        reason = item.get("reason")
        if not all(isinstance(value, str) and value.strip() for value in (rel, rule, reason)):
            raise AllowlistError("each allowlist entry requires path, rule, and reason")
        normalized = rel.replace("\\", "/").lstrip("./")
        if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise AllowlistError("allowlist paths must be safe relative paths")
        if rule not in {candidate.number for candidate in RULES}:
            raise AllowlistError("allowlist contains an unknown rule")
        result.add((normalized, rule))
    return result


def _is_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        chunk = handle.read(8192)
    return b"\x00" in chunk


class GitIgnore:
    """Small, deterministic matcher for this repository's simple ignore rules."""

    def __init__(self, patterns: list[tuple[bool, str, bool]]) -> None:
        self.patterns = patterns

    @classmethod
    def load(cls, path: Path) -> GitIgnore:
        patterns: list[tuple[bool, str, bool]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return cls(patterns)
        for raw in lines:
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            negated = value.startswith("!")
            if negated:
                value = value[1:]
            directory_only = value.endswith("/")
            value = value.strip("/").replace("\\", "/")
            if value:
                patterns.append((negated, value, directory_only))
        return cls(patterns)

    def matches(self, rel: str, *, is_dir: bool) -> bool:
        rel = rel.replace("\\", "/").strip("/")
        parts = rel.split("/") if rel else []
        ignored = False
        for negated, pattern, directory_only in self.patterns:
            if directory_only and not is_dir:
                continue
            if "/" in pattern:
                matched = fnmatch.fnmatchcase(rel, pattern)
            else:
                matched = any(fnmatch.fnmatchcase(part, pattern) for part in parts)
            if matched:
                ignored = not negated
        return ignored


def _walk_files(
    root: Path,
    *,
    relative_root: Path | None = None,
    gitignore: GitIgnore | None = None,
    forbidden: list[Finding] | None = None,
) -> Iterable[Path]:
    """Walk without following links, honoring local-artifact exclusions."""
    relative_root = relative_root or root
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(dirs):
            candidate = current_path / name
            rel = candidate.relative_to(relative_root).as_posix()
            if name.casefold() in FORBIDDEN_SOURCE_NAMES and forbidden is not None:
                forbidden.append(Finding(rel, 0, "R010", "forbidden-source-tree"))
                continue
            if name in SKIP_DIRS:
                continue
            if gitignore is not None and gitignore.matches(rel, is_dir=True):
                continue
            if candidate.is_symlink():
                yield candidate
                continue
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            candidate = current_path / name
            rel = candidate.relative_to(relative_root).as_posix()
            if gitignore is None or not gitignore.matches(rel, is_dir=False):
                yield candidate


def _scan_files(
    root: Path,
    paths: Iterable[Path],
    *,
    allowlist: set[tuple[str, str]],
    max_file_size: int,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(Finding(rel, 0, "R005", "symbolic-link"))
            continue
        try:
            if path.stat().st_size > max_file_size:
                findings.append(Finding(rel, 0, "R006", "unscanned-large-file"))
                continue
            if _is_binary(path):
                continue
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                for line_number, line in enumerate(handle, 1):
                    for rule in RULES:
                        if (rel, rule.number) in allowlist:
                            continue
                        if _matches_rule(rule, path, line):
                            findings.append(Finding(rel, line_number, rule.number, rule.category))
        except (OSError, UnicodeError):
            findings.append(Finding(rel, 0, "R007", "unreadable-file"))
    return findings


def scan_tree(
    root: Path,
    *,
    allowlist: set[tuple[str, str]] | None = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> list[Finding]:
    """Return sanitized findings for every readable text file in ``root``."""
    return _scan_files(
        root,
        _walk_files(root),
        allowlist=allowlist or set(),
        max_file_size=max_file_size,
    )


def scan_source_tree(
    root: Path,
    *,
    allowlist: set[tuple[str, str]] | None = None,
    max_file_size: int = MAX_FILE_SIZE,
) -> list[Finding]:
    """Scan the complete public source set rooted at ``root``."""
    findings: list[Finding] = []
    ignored = GitIgnore.load(root / ".gitignore")
    paths: list[Path] = []

    for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if child.name.casefold() in FORBIDDEN_SOURCE_NAMES:
            findings.append(Finding(child.name, 0, "R010", "forbidden-source-tree"))

    for entry_name in SOURCE_ENTRIES:
        entry = root / entry_name
        if not entry.exists() and not entry.is_symlink():
            continue
        if entry.is_file() or entry.is_symlink():
            if not ignored.matches(entry_name, is_dir=False):
                paths.append(entry)
            continue
        paths.extend(
            _walk_files(
                entry,
                relative_root=root,
                gitignore=ignored,
                forbidden=findings,
            )
        )

    findings.extend(
        _scan_files(
            root,
            paths,
            allowlist=allowlist or set(),
            max_file_size=max_file_size,
        )
    )
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan a release candidate or public source tree")
    parser.add_argument("legacy_root", nargs="?", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=None, help="Explicit scan root")
    parser.add_argument("--mode", choices=("release", "source"), default="release")
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--max-file-size", type=int, default=MAX_FILE_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root is not None and args.legacy_root is not None:
        print("privacy scan configuration error", file=sys.stderr)
        return 2
    selected_root = args.root or args.legacy_root
    if selected_root is None:
        selected_root = PROJECT_ROOT if args.mode == "source" else DEFAULT_RELEASE_ROOT
    root = selected_root.resolve()
    if not root.is_dir():
        print("privacy scan configuration error", file=sys.stderr)
        return 2
    try:
        allowed = load_allowlist(args.allowlist)
    except AllowlistError:
        print("privacy scan allowlist error", file=sys.stderr)
        return 2
    scanner = scan_source_tree if args.mode == "source" else scan_tree
    findings = scanner(root, allowlist=allowed, max_file_size=max(1, args.max_file_size))
    if findings:
        for finding in findings:
            print(f"{finding.path}:{finding.line}:{finding.rule}:{finding.category}")
        print(f"privacy scan failed: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("privacy scan passed: 0 findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
