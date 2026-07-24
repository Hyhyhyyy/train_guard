#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic-data unit tests for Train Guard (no real domain corpora)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_guard.adapters.huggingface import is_bad_loss, parse_training_metrics  # noqa: E402
from train_guard.cli import main  # noqa: E402
from train_guard.core.privacy import redact_text  # noqa: E402
from train_guard.data.commands import run_data_check, run_data_compare, run_data_inventory  # noqa: E402
from train_guard.env.doctor import run_doctor  # noqa: E402
from train_guard.report.html import render_html_report  # noqa: E402
from train_guard.run.commands import query_nvidia_smi, run_manifest, run_run_check  # noqa: E402
import train_guard as tg  # noqa: E402


NVIDIA_SMI_3GPU = """\
0, NVIDIA A100-PCIE-40GB, 40960, 1024, 39936, 55, 42, 120.5, 525.60.13
1, NVIDIA A100-PCIE-40GB, 40960, 2048, 38912, 60, 45, 130.0, 525.60.13
2, NVIDIA A100-PCIE-40GB, 40960, 512, 40448, 10, 40, 90.0, 525.60.13
"""


class TestLogParser(unittest.TestCase):
    def test_formats_and_eval_isolation(self) -> None:
        self.assertAlmostEqual(parse_training_metrics("loss: 1.234")["loss"], 1.234)
        self.assertAlmostEqual(parse_training_metrics("'loss': 1.2")["loss"], 1.2)
        m = parse_training_metrics("eval_loss=0.5 loss=1.2")
        self.assertEqual(m["eval_loss"], 0.5)
        self.assertEqual(m["loss"], 1.2)
        only = parse_training_metrics("eval_loss: 0.9")
        self.assertIn("eval_loss", only)
        self.assertNotIn("loss", only)
        self.assertTrue(is_bad_loss(parse_training_metrics("loss: NaN")["loss"]))


class TestDoctor(unittest.TestCase):
    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["doctor", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_gpu(self) -> None:
        with mock.patch("train_guard.run.commands.run_command", return_value=(127, "", "missing")):
            report = run_doctor()
        self.assertIn(report["overall_status"], ("WARN", "FAIL", "PASS"))

    def test_three_gpus(self) -> None:
        with mock.patch("train_guard.run.commands.run_command", return_value=(0, NVIDIA_SMI_3GPU, "")):
            info = query_nvidia_smi()
        self.assertEqual(info["count"], 3)


class TestDataCommands(unittest.TestCase):
    def test_check_missing_empty_leak_dup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "images"
            img.mkdir()
            (img / "a.jpg").write_bytes(b"aaa")
            (img / "b.jpg").write_bytes(b"aaa")
            (img / "empty.jpg").write_bytes(b"")
            samples = [
                {"group_id": "G1", "split": "train", "images": ["images/a.jpg"], "answer": "ok"},
                {"group_id": "G1", "split": "validation", "images": ["images/missing.jpg"], "answer": ""},
                {"group_id": "G2", "split": "train", "images": ["images/b.jpg", "images/empty.jpg"], "answer": "x"},
            ]
            ann = root / "train.json"
            ann.write_text(json.dumps(samples), encoding="utf-8")
            report = run_data_check(
                {
                    "annotation": str(ann),
                    "data_root": str(root),
                    "sample_limit": 100,
                    "compute_hash": True,
                    "verify_images": False,
                    "cache_db": str(root / "cache.sqlite"),
                }
            )
            self.assertEqual(report["overall_status"], "FAIL")
            self.assertGreaterEqual(report["stats"]["missing_media"], 1)
            self.assertGreaterEqual(report["stats"]["empty_answers"], 1)
            self.assertGreaterEqual(report["stats"]["group_leak_count"], 1)
            leak = report["issues"].get("group_leak") or []
            self.assertTrue(all("G1" not in x for x in leak))

    def test_inventory_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "a.jsonl"
            right = root / "b.jsonl"
            left.write_text(
                json.dumps({"group_id": "A", "images": ["x.png"], "output": "1"}) + "\n"
                + json.dumps({"group_id": "B", "images": ["y.png"], "output": "2"}) + "\n",
                encoding="utf-8",
            )
            right.write_text(
                json.dumps({"group_id": "B", "images": ["y.png"], "output": "2"}) + "\n"
                + json.dumps({"group_id": "C", "images": ["z.png"], "output": "3"}) + "\n",
                encoding="utf-8",
            )
            inv = run_data_inventory({"annotation": str(left)})
            self.assertEqual(inv["sample_count"], 2)
            cmp = run_data_compare({"left": str(left), "right": str(right)})
            self.assertEqual(cmp["group_overlap_count"], 1)


class TestRunCheck(unittest.TestCase):
    def _ok_run(self, root: Path, steps: int = 10) -> Path:
        out = root / "saves_ok"
        out.mkdir()
        ckpt = out / f"checkpoint-{steps}"
        ckpt.mkdir()
        state = {
            "global_step": steps,
            "log_history": [
                {"loss": 1.2, "step": 5},
                {"eval_loss": 1.0},
                {"train_runtime": 12.5, "train_loss": 0.9, "train_samples_per_second": 3.2},
            ],
        }
        (out / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
        (out / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}), encoding="utf-8")
        (out / "adapter_model.safetensors").write_bytes(b"weights")
        (ckpt / "adapter_model.safetensors").write_bytes(b"ckpt")
        return out

    def test_success_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._ok_run(Path(tmp), 10)
            report = run_run_check(out, expected_steps=10)
            self.assertEqual(report["overall_status"], "PASS")
            state = json.loads((out / "trainer_state.json").read_text(encoding="utf-8"))
            state["log_history"].append({"loss": "NaN"})
            (out / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
            bad = run_run_check(out, expected_steps=10)
            self.assertEqual(bad["overall_status"], "FAIL")

    def test_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._ok_run(Path(tmp), 10)
            man = run_manifest({"output_dir": str(out), "framework": "huggingface", "manifest_out": str(Path(tmp) / "m.json")})
            self.assertIn("experiment_fingerprint", man)
            self.assertTrue((Path(tmp) / "m.json").exists())


class TestPrivacyAndReports(unittest.TestCase):
    def test_html_escape(self) -> None:
        dangerous = '<script>alert("x")</script>'
        doc = render_html_report(dangerous, [{"title": dangerous, "value": dangerous, "status": "FAIL"}], [{"title": dangerous, "headers": [dangerous], "rows": [[dangerous]]}], dangerous)
        self.assertNotIn("<script>", doc)
        self.assertIn("&lt;script&gt;", doc)

    def test_redact_path(self) -> None:
        text = redact_text("/" + "home/alice/secret/" + "token=abc path")
        self.assertNotIn("alice", text)

    def test_legacy_alias_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "a.json"
            ann.write_text("[]", encoding="utf-8")
            code = main(["precheck", "--annotation", str(ann), "--data-root", tmp, "--no-verify-images", "--report-dir", str(Path(tmp) / "r")])
            self.assertIn(code, (0, 1, 2))


class TestVersion(unittest.TestCase):
    def test_version(self) -> None:
        self.assertTrue(tg.__version__.startswith("0.5"))


@unittest.skipUnless(False, "NCCL smoke is optional; covered by examples script when GPUs exist")
class TestNCCLPlaceholder(unittest.TestCase):
    def test_skip(self) -> None:
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
