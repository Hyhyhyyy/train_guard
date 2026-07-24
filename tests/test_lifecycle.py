#!/usr/bin/env python3
"""Lifecycle schema and run-loop integration tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_guard.adapters.huggingface import LLaMAFactoryFrameworkAdapter  # noqa: E402
from train_guard.run.commands import cmd_run_watch, run_manifest, run_run_check, run_run_compare  # noqa: E402
from train_guard.run.lifecycle import (  # noqa: E402
    PHASE_ABORTED,
    PHASE_FINISHED,
    append_lifecycle_event,
    lifecycle_path,
    make_lifecycle_event,
    summarize_lifecycle,
)


class TestLifecycleSchema(unittest.TestCase):
    def test_summarize_start_heartbeat_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = lifecycle_path(Path(tmp))
            append_lifecycle_event(path, make_lifecycle_event("start", framework="huggingface", global_step=0))
            append_lifecycle_event(
                path,
                make_lifecycle_event(
                    "checkpoint",
                    framework="huggingface",
                    global_step=5,
                    checkpoints=["checkpoint-5"],
                ),
            )
            append_lifecycle_event(
                path,
                make_lifecycle_event("heartbeat", framework="huggingface", global_step=5),
            )
            append_lifecycle_event(
                path,
                make_lifecycle_event("finish", framework="huggingface", global_step=10, checkpoints=["checkpoint-5", "checkpoint-10"]),
            )
            summary = summarize_lifecycle(path)
            self.assertEqual(summary["phase"], PHASE_FINISHED)
            self.assertTrue(summary["has_finish"])
            self.assertEqual(summary["global_step"], 10)
            self.assertEqual(summary["event_count"], 4)
            self.assertIn("checkpoint-10", summary["checkpoints"])

    def test_abort_wins_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = lifecycle_path(Path(tmp))
            append_lifecycle_event(path, make_lifecycle_event("start"))
            append_lifecycle_event(path, make_lifecycle_event("abort", message="pid dead"))
            summary = summarize_lifecycle(path)
            self.assertEqual(summary["phase"], PHASE_ABORTED)
            self.assertTrue(summary["has_abort"])


class TestWatchLifecycle(unittest.TestCase):
    def test_watch_once_writes_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            with mock.patch("train_guard.run.commands.query_nvidia_smi", return_value={"ok": False, "available": False, "gpus": [], "count": 0, "error": "none", "driver_version": None}):
                code = cmd_run_watch({
                    "once": True,
                    "interval": 1,
                    "output_dir": str(out),
                    "framework": "huggingface",
                })
            self.assertEqual(code, 0)
            life = lifecycle_path(out)
            self.assertTrue(life.is_file())
            summary = summarize_lifecycle(life)
            self.assertEqual(summary["phase"], PHASE_FINISHED)
            self.assertGreaterEqual(summary["event_count"], 2)
            report = run_run_check(out, framework="huggingface")
            names = {c["name"]: c["status"] for c in report["checks"]}
            self.assertEqual(names.get("lifecycle"), "PASS")


class TestManifestCompareLifecycle(unittest.TestCase):
    def _seed(self, root: Path) -> Path:
        out = root / "saves"
        out.mkdir(parents=True)
        (out / "trainer_state.json").write_text(json.dumps({"global_step": 3, "log_history": [{"loss": 1.0}]}), encoding="utf-8")
        (out / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}), encoding="utf-8")
        (out / "adapter_model.safetensors").write_bytes(b"w")
        ckpt = out / "checkpoint-3"
        ckpt.mkdir()
        (ckpt / "adapter_model.safetensors").write_bytes(b"c")
        path = lifecycle_path(out)
        append_lifecycle_event(path, make_lifecycle_event("start", global_step=0))
        append_lifecycle_event(path, make_lifecycle_event("finish", global_step=3, checkpoints=["checkpoint-3"]))
        return out

    def test_manifest_includes_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._seed(Path(tmp))
            man = run_manifest({"output_dir": str(out), "framework": "huggingface"})
            self.assertEqual(man["lifecycle"]["phase"], PHASE_FINISHED)
            self.assertIn("experiment_fingerprint", man)

    def test_compare_includes_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = self._seed(root / "left")
            right = self._seed(root / "right")
            report = run_run_compare(left, right, framework="huggingface")
            self.assertEqual(report["left"]["lifecycle_phase"], PHASE_FINISHED)
            self.assertEqual(report["right"]["lifecycle_phase"], PHASE_FINISHED)


class TestLLaMAFactoryNested(unittest.TestCase):
    def test_nested_checkpoint_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "saves"
            nested = root / "example" / "lora"
            ckpt = nested / "checkpoint-2"
            ckpt.mkdir(parents=True)
            (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 2}), encoding="utf-8")
            (ckpt / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}), encoding="utf-8")
            (ckpt / "adapter_model.safetensors").write_bytes(b"x")
            adapter = LLaMAFactoryFrameworkAdapter()
            state = adapter.locate_trainer_state(root)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertTrue(state.is_file())
            arts = adapter.find_adapter_artifacts(root)
            self.assertTrue(arts.adapter_configs)
            self.assertTrue(arts.checkpoints)


if __name__ == "__main__":
    unittest.main()
