#!/usr/bin/env python3
"""Release privacy-gate tests using synthetic content only."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "privacy_scan.py"


class TestPrivacyScanner(unittest.TestCase):
    def init_repo(self, tree: Path, *paths: str, force: bool = False) -> None:
        subprocess.run(["git", "init", "-q", str(tree)], check=True, capture_output=True)
        command = ["git", "-C", str(tree), "add"]
        if force:
            command.append("-f")
        command.extend(paths)
        subprocess.run(command, check=True, capture_output=True)

    def run_scan(self, tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCANNER), str(tree), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def run_source_scan(self, tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCANNER), "--mode", "source", "--root", str(tree), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_rules_exit_code_and_no_sensitive_echo(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            secret_value = "s3cr3t-value-never-echo"
            person_value = "Example Person Never Echo"
            domain_terms = "pat" + "ient diag" + "nosis"
            person_field = "full_" + "name"
            user_path = "/" + "root/hidden/input.json"
            (tree / "sample.txt").write_text(
                "Token="
                + secret_value
                + "\n"
                + user_path
                + "\n"
                + domain_terms
                + "\n"
                + person_field
                + "="
                + person_value
                + "\n",
                encoding="utf-8",
            )
            result = self.run_scan(tree)
            self.assertEqual(result.returncode, 1)
            self.assertIn("sample.txt:1:R001:credential", result.stdout)
            self.assertIn("sample.txt:2:R002:absolute-path", result.stdout)
            self.assertIn("sample.txt:3:R003:domain-specific", result.stdout)
            self.assertIn("sample.txt:4:R004:personal-data", result.stdout)
            self.assertNotIn(secret_value, result.stdout + result.stderr)
            self.assertNotIn(person_value, result.stdout + result.stderr)
            self.assertNotIn("diag" + "nosis", result.stdout + result.stderr)

    def test_source_mode_scans_only_git_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / ".gitignore").write_text("*.log\ncache/\n", encoding="utf-8")
            (tree / "README.md").write_text("Public project\n", encoding="utf-8")
            (tree / "src").mkdir()
            (tree / "src" / "clean.py").write_text("value = 1\n", encoding="utf-8")
            credential_key = "tok" + "en"
            (tree / "src" / "ignored.log").write_text(
                credential_key + "=hidden-log-value\n", encoding="utf-8"
            )
            (tree / "src" / "cache").mkdir()
            (tree / "src" / "cache" / "ignored.txt").write_text(
                credential_key + "=hidden-cache-value\n", encoding="utf-8"
            )
            (tree / "outside.txt").write_text(
                credential_key + "=not-public-value\n", encoding="utf-8"
            )
            self.init_repo(tree, ".gitignore", "README.md", "src/clean.py")
            result = self.run_source_scan(tree)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("candidates=3", result.stdout)

            (tree / "scripts").mkdir()
            address = ".".join(("10", "20", "30", "40"))
            (tree / "scripts" / "unsafe.py").write_text(
                'server = "' + address + '"\n', encoding="utf-8"
            )
            subprocess.run(
                ["git", "-C", str(tree), "add", "scripts/unsafe.py"],
                check=True,
                capture_output=True,
            )
            blocked = self.run_source_scan(tree)
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("scripts/unsafe.py:1:R009:server-address", blocked.stdout)
            self.assertNotIn(address, blocked.stdout + blocked.stderr)

    def test_source_mode_blocks_tracked_high_risk_trees(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / ".gitignore").write_text("legacy/\nprivate/\nreports/\n", encoding="utf-8")
            (tree / "README.md").write_text("Public project\n", encoding="utf-8")
            for name in ("legacy", "private", "reports"):
                (tree / name).mkdir()
                (tree / name / "fixture.txt").write_text("synthetic\n", encoding="utf-8")
            self.init_repo(
                tree,
                ".gitignore",
                "README.md",
                "legacy/fixture.txt",
                "private/fixture.txt",
                "reports/fixture.txt",
                force=True,
            )
            result = self.run_source_scan(tree)
            self.assertEqual(result.returncode, 1)
            for name in ("legacy", "private", "reports"):
                self.assertIn(f"{name}/fixture.txt:0:R010:forbidden-source-tree", result.stdout)

    def test_private_key_is_reported_without_echo(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            marker = "-----BEGIN " + "PRIVATE KEY-----"
            (tree / "key.txt").write_text(marker + "\n", encoding="utf-8")
            result = self.run_scan(tree)
            self.assertEqual(result.returncode, 1)
            self.assertIn("key.txt:1:R008:private-key", result.stdout)
            self.assertNotIn(marker, result.stdout + result.stderr)

    def test_loopback_address_is_allowed_but_remote_address_is_not(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / "local.txt").write_text(
                'server = "127.0.0.1"\nendpoint = "localhost"\n',
                encoding="utf-8",
            )
            clean = self.run_scan(tree)
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)

            address = ".".join(("10", "20", "30", "40"))
            (tree / "remote.txt").write_text(address + "\n", encoding="utf-8")
            blocked = self.run_scan(tree)
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("remote.txt:1:R009:server-address", blocked.stdout)
            self.assertNotIn(address, blocked.stdout + blocked.stderr)

    def test_allowlist_requires_reason_and_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / "compat.txt").write_text("pat" + "ient\n", encoding="utf-8")
            allow = tree / "allow.json"
            allow.write_text(
                json.dumps(
                    [
                        {
                            "path": "compat.txt",
                            "rule": "R003",
                            "reason": "Deprecated compatibility fixture.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = self.run_scan(tree, "--allowlist", str(allow))
            self.assertEqual(result.returncode, 0)

            allow.write_text(json.dumps([{"path": "compat.txt", "rule": "R003"}]), encoding="utf-8")
            bad = self.run_scan(tree, "--allowlist", str(allow))
            self.assertEqual(bad.returncode, 2)

    def test_allowlist_preserves_dot_prefixed_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            workflow = tree / ".github" / "workflows" / "publish.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("permissions:\n  id-token: write\n", encoding="utf-8")
            allow = tree / "allow.json"
            allow.write_text(
                json.dumps(
                    [
                        {
                            "path": ".github/workflows/publish.yml",
                            "rule": "R001",
                            "reason": "Synthetic OIDC permission fixture.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.init_repo(tree, ".github/workflows/publish.yml")
            result = self.run_scan(
                tree,
                "--mode",
                "source",
                "--allowlist",
                str(allow),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_binary_requires_an_exact_allowlist_entry(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / "asset.bin").write_bytes(b"\x00access_" + b"tok" + b"en=hidden-binary-value")
            result = self.run_scan(tree)
            self.assertEqual(result.returncode, 1)
            self.assertIn("asset.bin:0:R011:binary-file", result.stdout)
            self.assertNotIn("hidden-binary-value", result.stdout + result.stderr)

            allow = tree / "allow.json"
            allow.write_text(
                json.dumps(
                    [
                        {
                            "path": "asset.bin",
                            "rule": "R011",
                            "reason": "Synthetic binary fixture.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            allowed = self.run_scan(tree, "--allowlist", str(allow))
            self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)
            self.assertIn("allowed_binary=1", allowed.stdout)

            (tree / "other.bin").write_bytes(b"synthetic" * 1200 + b"\x00")
            blocked = self.run_scan(tree, "--allowlist", str(allow))
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("other.bin:0:R011:binary-file", blocked.stdout)

    def test_large_file_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / "large.txt").write_text("0123456789", encoding="utf-8")
            result = self.run_scan(tree, "--max-file-size", "4")
            self.assertEqual(result.returncode, 1)
            self.assertIn("large.txt:0:R006:unscanned-large-file", result.stdout)

    def test_python_annotations_and_loader_fullname_are_not_findings(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            joined_name = "full" + "name"
            (tree / "loader.py").write_text(
                f"def load(token: str, {joined_name}: str) -> None:\n"
                f"    module_name = {joined_name}\n"
                f"    self.{joined_name} = module_name\n"
                "BUNDLED = {'module': 'def redact(token: str) -> str: ...'}\n",
                encoding="utf-8",
            )
            result = self.run_scan(tree)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_token_and_fullname_values_remain_findings(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            secret_value = "real-secret-never-echo"
            person_value = "Real Person Never Echo"
            credential_key = "tok" + "en"
            person_field = "full" + "name"
            (tree / "config.py").write_text(
                credential_key
                + ' = "'
                + secret_value
                + '"\n'
                + 'record = {"'
                + person_field
                + '": "'
                + person_value
                + '"}\n',
                encoding="utf-8",
            )
            result = self.run_scan(tree)
            self.assertEqual(result.returncode, 1)
            self.assertIn("config.py:1:R001:credential", result.stdout)
            self.assertIn("config.py:2:R004:personal-data", result.stdout)
            self.assertNotIn(secret_value, result.stdout + result.stderr)
            self.assertNotIn(person_value, result.stdout + result.stderr)

    def test_bundled_newline_is_not_a_windows_path_but_user_path_is(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tree = Path(tmp)
            (tree / "bundle.py").write_text(
                "MODULE = {'source': 'if m:\\n    return m'}\n", encoding="utf-8"
            )
            clean = self.run_scan(tree)
            self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)

            user_path = "C:" + "\\Users\\ExampleUser\\private.txt"
            (tree / "path.txt").write_text(user_path + "\n", encoding="utf-8")
            blocked = self.run_scan(tree)
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("path.txt:1:R002:absolute-path", blocked.stdout)
            self.assertNotIn(user_path, blocked.stdout + blocked.stderr)


if __name__ == "__main__":
    unittest.main()
