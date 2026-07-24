#!/usr/bin/env python3
"""Candidate release manifest policy tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release_manifest.py"
SPEC = importlib.util.spec_from_file_location("check_release_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestReleaseManifest(unittest.TestCase):
    def test_clean_candidate_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / "train_guard.py").write_text("print('ok')\n", encoding="utf-8")
            files, rejected = MODULE.validate_manifest(tree)
            self.assertEqual(files, ["train_guard.py"])
            self.assertEqual(rejected, [])

    def test_rejects_sensitive_and_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            for directory, filename in (
                ("private", "input.json"),
                ("reports", "run.json"),
                ("__pycache__", "module.pyc"),
            ):
                target = tree / directory
                target.mkdir()
                (target / filename).write_bytes(b"fixture")
            (tree / "run.log").write_text("fixture", encoding="utf-8")
            (tree / "cache.sqlite3").write_bytes(b"fixture")
            _, rejected = MODULE.validate_manifest(tree)
            self.assertTrue(any(path.startswith("private/") for path in rejected))
            self.assertTrue(any(path.startswith("reports/") for path in rejected))
            self.assertTrue(any("__pycache__" in path for path in rejected))
            self.assertIn("run.log", rejected)
            self.assertIn("cache.sqlite3", rejected)

    def test_fixed_allowlist_and_sha256_detect_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            candidate = tree / "train_guard.py"
            candidate.write_text("print('ok')\n", encoding="utf-8")
            files, rejected = MODULE.inventory(tree)
            self.assertEqual(rejected, [])
            manifest = tree / "manifest.json"
            MODULE.write_manifest(manifest, files)
            self.assertEqual(MODULE.load_manifest(manifest), files)

            candidate.write_text("print('changed')\n", encoding="utf-8")
            changed, _ = MODULE.inventory(tree)
            self.assertNotEqual(changed["train_guard.py"], files["train_guard.py"])

    def test_manifest_rejects_unsafe_paths_and_bad_hashes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            manifest = Path(tmp) / "manifest.json"
            for files in ({"../escape": "0" * 64}, {"safe.py": "not-a-sha256"}):
                manifest.write_text(
                    json.dumps({"schema_version": 1, "algorithm": "sha256", "files": files}),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    MODULE.load_manifest(manifest)

            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "algorithm": "sha512",
                        "files": {"safe.py": "0" * 64},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                MODULE.load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
