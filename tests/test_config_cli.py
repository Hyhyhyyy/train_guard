#!/usr/bin/env python3
"""Configuration and initialization tests using synthetic paths only."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_guard.cli import main  # noqa: E402
from train_guard.core.config import (  # noqa: E402
    ConfigError,
    config_template,
    load_config_file,
    resolve_command_config,
    validate_config,
    write_config_template,
)


class TestInit(unittest.TestCase):
    def test_init_templates_are_json_and_domain_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("generic", "transformers", "llamafactory"):
                output = root / f"{name}.json"
                with redirect_stdout(StringIO()):
                    code = main(["init", "--template", name, "--output", str(output)])
                self.assertEqual(code, 0)
                data = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(data["schema_version"], 1)
                check = data["data"]["check"]
                self.assertEqual(check["group_id"], "group_id")
                self.assertEqual(check["messages"], "messages")
                self.assertEqual(check["media"], "media")
                self.assertNotIn("images_field", check)
                self.assertNotIn("answer_field", check)
                self.assertIn("inventory", data["data"])
                self.assertIn("compare", data["data"])
                self.assertIn("check", data["run"])
                self.assertIn("compare", data["run"])
                self.assertIn("manifest", data)
                self.assertEqual(
                    data["run"]["check"]["framework"], data["run"]["watch"]["framework"]
                )

    def test_init_refuses_then_force_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "config.json"
            output.write_text('{"keep": true}\n', encoding="utf-8")
            with redirect_stderr(StringIO()):
                refused = main(["init", "--output", str(output)])
            self.assertEqual(refused, 6)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"keep": True})
            with redirect_stdout(StringIO()):
                forced = main(["init", "--output", str(output), "--force"])
            self.assertEqual(forced, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], 1)

    def test_yaml_missing_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "config.yaml"
            with mock.patch("train_guard.core.config.try_import_yaml", return_value=None):
                with self.assertRaises(ConfigError) as ctx:
                    write_config_template(output, "generic")
            message = str(ctx.exception)
            self.assertIn("Problem:", message)
            self.assertIn("Location:", message)
            self.assertIn("Fix:", message)
            self.assertFalse(output.exists())


class TestConfigSchema(unittest.TestCase):
    def _write(self, root: Path, data: object, name: str = "config.json") -> Path:
        path = root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_cli_beats_file_beats_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write(
                root,
                {
                    "schema_version": 1,
                    "run": {"watch": {"interval": 17, "framework": "generic"}},
                },
            )
            file_result = resolve_command_config(("run", "watch"), path, {})
            self.assertEqual(file_result["interval"], 17)
            self.assertEqual(file_result["disk_free_gb_threshold"], 10.0)
            cli_result = resolve_command_config(("run", "watch"), path, {"interval": 3})
            self.assertEqual(cli_result["interval"], 3)

    def test_schema_version_type_and_unknown_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases: tuple[tuple[dict[str, object], str], ...] = (
                ({"doctor": {}}, "schema_version"),
                ({"schema_version": "1"}, "must be an integer"),
                ({"schema_version": 99}, "not supported"),
                ({"schema_version": 1, "doctor": {"mystery": True}}, "unknown field"),
                ({"schema_version": 1, "doctor": {"expected_gpus": "two"}}, "expected"),
            )
            for index, (data, expected) in enumerate(cases):
                path = self._write(root, data, f"{index}.json")
                with self.assertRaises(ConfigError) as ctx:
                    validate_config(load_config_file(path), path)
                text = str(ctx.exception)
                self.assertIn(expected, text)
                self.assertIn("Location:", text)
                self.assertIn("Fix:", text)

    def test_relative_paths_use_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "nested"
            cfg_dir.mkdir()
            path = self._write(
                cfg_dir,
                {
                    "schema_version": 1,
                    "data": {"check": {"annotation": "../data/train.jsonl"}},
                },
            )
            result = resolve_command_config(("data", "check"), path, {})
            self.assertEqual(
                Path(result["annotation"]),
                (cfg_dir / "../data/train.jsonl").resolve(),
            )
            cli_path = "./cli/train.jsonl"
            overridden = resolve_command_config(("data", "check"), path, {"annotation": cli_path})
            self.assertEqual(overridden["annotation"], cli_path)

    def test_windows_and_linux_absolute_paths_are_not_rebased(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = ("C:\\data\\train.jsonl", "/var/tmp/train.jsonl")
            for index, value in enumerate(values):
                path = self._write(
                    root,
                    {
                        "schema_version": 1,
                        "data": {"check": {"annotation": value}},
                    },
                    f"path-{index}.json",
                )
                result = resolve_command_config(("data", "check"), path, {})
                self.assertEqual(result["annotation"], value)

    def test_yaml_input_missing_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("schema_version: 1\n", encoding="utf-8")
            with mock.patch("train_guard.core.config.try_import_yaml", return_value=None):
                with self.assertRaises(ConfigError) as ctx:
                    load_config_file(path)
            self.assertIn("PyYAML", str(ctx.exception))

    def test_invalid_config_stops_before_data_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                {
                    "schema_version": 1,
                    "data": {"check": {"annotation": "./missing.jsonl", "unknown": 1}},
                },
            )
            stderr = StringIO()
            with mock.patch("train_guard.cli.run_data_check") as data_check:
                with redirect_stderr(stderr):
                    code = main(["data", "check", "--config", str(path)])
            self.assertEqual(code, 4)
            data_check.assert_not_called()
            self.assertIn("Location:", stderr.getvalue())
            self.assertIn("Fix:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
