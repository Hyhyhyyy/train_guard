#!/usr/bin/env python3
"""Privacy gate for a complete candidate release tree.

Findings deliberately contain metadata only. Matched text is never printed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "release"
MAX_FILE_SIZE = 8 * 1024 * 1024
FORBIDDEN_SOURCE_NAMES = {"legacy", "private", "reports"}
POLICY_RULES = {"R005", "R006", "R007", "R010", "R011"}


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


@dataclass
class ScanStats:
    candidates: int = 0
    text_files: int = 0
    allowed_binary_files: int = 0


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
        re.compile(
            r"(?i)\b(?:full[_ -]?name|date[_ -]?of[_ -]?birth|social[_ -]?security)\b|姓名|出生日期|身份证"
        ),
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
            r"|(?<![\d.])(?!127\.0\.0\.1(?::\d{1,5})?(?![\d.]))"
            r"(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?![\d.])"
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
            stripped.startswith('r"') or 're.compile(r"' in stripped
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
        if (
            not isinstance(rel, str)
            or not rel.strip()
            or not isinstance(rule, str)
            or not rule.strip()
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise AllowlistError("each allowlist entry requires path, rule, and reason")
        normalized = rel.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise AllowlistError("allowlist paths must be safe relative paths")
        if rule not in ({candidate.number for candidate in RULES} | POLICY_RULES):
            raise AllowlistError("allowlist contains an unknown rule")
        result.add((normalized, rule))
    return result


def _is_binary(path: Path) -> bool:
    """Classify binary data deterministically without relying on extensions."""
    content = path.read_bytes()
    if b"\x00" in content:
        return True
    try:
        content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return True
    return False


def _walk_files(
    root: Path,
) -> Iterable[Path]:
    """Walk a release candidate without following directory links."""
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(dirs):
            candidate = current_path / name
            if candidate.is_symlink():
                yield candidate
                continue
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            yield current_path / name


def _git_tracked_files(root: Path) -> list[Path]:
    """Return the repository's exact tracked-file boundary."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("git tracked-file enumeration failed") from exc
    paths: list[Path] = []
    for raw in result.stdout.split(b"\x00"):
        if not raw:
            continue
        try:
            rel = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("git returned a non-UTF-8 path") from exc
        candidate = root / rel
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("git returned an unsafe path") from exc
        if not candidate.exists() and not candidate.is_symlink():
            continue
        paths.append(candidate)
    return paths


def _scan_files(
    root: Path,
    paths: Iterable[Path],
    *,
    allowlist: set[tuple[str, str]],
    max_file_size: int,
    stats: ScanStats,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        stats.candidates += 1
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(Finding(rel, 0, "R005", "symbolic-link"))
            continue
        try:
            if path.stat().st_size > max_file_size:
                findings.append(Finding(rel, 0, "R006", "unscanned-large-file"))
                continue
            if _is_binary(path):
                if (rel, "R011") in allowlist:
                    stats.allowed_binary_files += 1
                else:
                    findings.append(Finding(rel, 0, "R011", "binary-file"))
                continue
            stats.text_files += 1
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
    stats: ScanStats | None = None,
) -> list[Finding]:
    """Return sanitized findings for every readable text file in ``root``."""
    scan_stats = stats or ScanStats()
    return _scan_files(
        root,
        _walk_files(root),
        allowlist=allowlist or set(),
        max_file_size=max_file_size,
        stats=scan_stats,
    )


def scan_source_tree(
    root: Path,
    *,
    allowlist: set[tuple[str, str]] | None = None,
    max_file_size: int = MAX_FILE_SIZE,
    stats: ScanStats | None = None,
) -> list[Finding]:
    """Scan exactly the files tracked by Git at ``root``."""
    findings: list[Finding] = []
    paths = _git_tracked_files(root)
    scan_stats = stats or ScanStats()
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if any(part.casefold() in FORBIDDEN_SOURCE_NAMES for part in Path(rel).parts):
            findings.append(Finding(rel, 0, "R010", "forbidden-source-tree"))

    findings.extend(
        _scan_files(
            root,
            paths,
            allowlist=allowlist or set(),
            max_file_size=max_file_size,
            stats=scan_stats,
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
    stats = ScanStats()
    try:
        findings = scanner(
            root,
            allowlist=allowed,
            max_file_size=max(1, args.max_file_size),
            stats=stats,
        )
    except RuntimeError:
        print("privacy scan git boundary error", file=sys.stderr)
        return 2
    print(
        "privacy scan stats: "
        f"candidates={stats.candidates} text={stats.text_files} "
        f"allowed_binary={stats.allowed_binary_files} findings={len(findings)}"
    )
    if findings:
        for finding in findings:
            print(f"{finding.path}:{finding.line}:{finding.rule}:{finding.category}")
        print(f"privacy scan failed: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("privacy scan passed: 0 findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
