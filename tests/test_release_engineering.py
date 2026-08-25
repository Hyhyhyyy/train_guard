#!/usr/bin/env python3
"""Documentation and built-sdist release contracts."""

from __future__ import annotations

import tarfile
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


class TestReleaseEngineering(unittest.TestCase):
    def test_readmes_share_release_contracts(self) -> None:
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README_zh-CN.md").read_text(encoding="utf-8")
        shared = (
            "0.6.0rc1",
            "git clone https://github.com/Hyhyhyyy/train_guard.git",
            'python -m pip install -e ".[all]"',
            "train-guard run supervise",
            "train-guard show",
            "docs/CLI.md",
            "docs/CONFIGURATION.md",
            "docs/RELIABILITY.md",
            "docs/RELEASE.md",
        )
        for marker in shared:
            self.assertIn(marker, english)
            self.assertIn(marker, chinese)

    def test_built_sdist_has_explicit_boundary(self) -> None:
        archives = sorted((ROOT / "dist").glob("*.tar.gz"))
        if not archives:
            self.skipTest("build the sdist before checking its archive boundary")
        with tarfile.open(archives[-1], "r:gz") as archive:
            members = {
                PurePosixPath(member.name).relative_to(*PurePosixPath(member.name).parts[:1])
                for member in archive.getmembers()
                if member.isfile()
            }
        required = {
            PurePosixPath("README.md"),
            PurePosixPath("docs/RELEASE.md"),
            PurePosixPath("configs/release_manifest.json"),
            PurePosixPath("scripts/check_release_manifest.py"),
            PurePosixPath("src/train_guard/__init__.py"),
        }
        self.assertTrue(required <= members)
        forbidden_roots = {".github", "benchmarks", "release", "tests"}
        self.assertFalse(any(path.parts and path.parts[0] in forbidden_roots for path in members))


if __name__ == "__main__":
    unittest.main()
