#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""monitor 子命令与日志解析相关测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_guard as tg  # noqa: E402


NVIDIA_SMI_3GPU = """\
0, NVIDIA A100-PCIE-40GB, 40960, 1024, 39936, 55, 42, 120.5, 525.60.13
1, NVIDIA A100-PCIE-40GB, 40960, 2048, 38912, 60, 45, 130.0, 525.60.13
2, NVIDIA A100-PCIE-40GB, 40960, 512, 40448, 10, 40, 90.0, 525.60.13
"""


class TestLogParser(unittest.TestCase):
    """日志指标解析。"""

    def test_basic_formats(self) -> None:
        cases = [
            ("loss: 1.234", {"loss": 1.234}),
            ("'loss': 1.234", {"loss": 1.234}),
            ('"loss": 1.234', {"loss": 1.234}),
            ("eval_loss=1.234", {"eval_loss": 1.234}),
            ("learning_rate: 1e-5", {"learning_rate": 1e-5}),
            ("grad_norm: 2.5", {"grad_norm": 2.5}),
            ("epoch: 0.5", {"epoch": 0.5}),
            ("step: 100", {"step": 100.0}),
        ]
        for line, expected in cases:
            with self.subTest(line=line):
                got = tg.parse_training_metrics(line)
                for k, v in expected.items():
                    self.assertIn(k, got)
                    self.assertAlmostEqual(got[k], v, places=8)

    def test_eval_loss_not_as_loss(self) -> None:
        got = tg.parse_training_metrics("{'eval_loss': 0.5, 'loss': 1.2}")
        self.assertEqual(got.get("eval_loss"), 0.5)
        self.assertEqual(got.get("loss"), 1.2)
        only_eval = tg.parse_training_metrics("eval_loss: 0.9")
        self.assertIn("eval_loss", only_eval)
        self.assertNotIn("loss", only_eval)

    def test_nan_inf(self) -> None:
        self.assertTrue(tg.is_bad_loss(float("nan")))
        self.assertTrue(tg.is_bad_loss(float("inf")))
        nan_metrics = tg.parse_training_metrics("loss: NaN")
        self.assertTrue(tg.is_bad_loss(nan_metrics["loss"]))
        inf_metrics = tg.parse_training_metrics("loss: Infinity")
        self.assertTrue(tg.is_bad_loss(inf_metrics["loss"]))


class TestMonitor(unittest.TestCase):
    """监控采样测试。"""

    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            tg.main(["monitor", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_once_and_stale_log_and_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "train.log"
            log_file.write_text("loss: 1.0\neval_loss: 0.8\n", encoding="utf-8")
            # 人为设置过期
            old = time.time() - 60 * 60
            import os

            os.utime(log_file, (old, old))

            out_dir = tmp_path / "mon"
            state: dict = {"idle_counts": {}, "log_offset": 0, "last_metrics_fp": None}
            cfg = {
                "log_file": str(log_file),
                "output_dir": str(out_dir),
                "stale_log_minutes": 10,
                "disk_free_gb_threshold": 10**9,  # 极高阈值，触发磁盘告警
                "expected_gpu_count": 3,
                "idle_gpu_util_threshold": 5.0,
                "idle_gpu_consecutive": 3,
            }

            with mock.patch.object(tg, "run_command", return_value=(0, NVIDIA_SMI_3GPU, "")):
                # 第一次：日志过期但 offset=0 会读到旧内容；先重置 mtime 再读指标
                sample = tg.collect_monitor_sample(cfg, state)
            codes = {a["code"] for a in sample["alerts"]}
            self.assertIn("stale_log", codes)
            self.assertIn("disk_low", codes)

            # NaN 告警
            log_file.write_text("loss: nan\n", encoding="utf-8")
            state["log_offset"] = 0
            state["last_metrics_fp"] = None
            # 刷新 mtime
            now = time.time()
            os.utime(log_file, (now, now))
            cfg["disk_free_gb_threshold"] = 0.0
            cfg["stale_log_minutes"] = 9999
            with mock.patch.object(tg, "run_command", return_value=(0, NVIDIA_SMI_3GPU, "")):
                sample2 = tg.collect_monitor_sample(cfg, state)
            codes2 = {a["code"] for a in sample2["alerts"]}
            self.assertIn("loss_nan_inf", codes2)
            self.assertTrue(tg.is_bad_loss(sample2["metrics"]["loss"]))

    def test_trainer_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            ckpt = out_dir / "checkpoint-10"
            ckpt.mkdir()
            ts = ckpt / "trainer_state.json"
            ts.write_text(
                json.dumps(
                    {
                        "global_step": 10,
                        "epoch": 0.2,
                        "log_history": [
                            {"loss": 2.0, "learning_rate": 1e-5, "epoch": 0.1},
                            {"eval_loss": 1.5, "epoch": 0.2},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            info = tg.read_trainer_state(ts)
            self.assertTrue(info["ok"])
            self.assertEqual(info["global_step"], 10)
            self.assertEqual(info["latest"]["eval_loss"], 1.5)

            state: dict = {"idle_counts": {}, "log_offset": 0, "last_metrics_fp": None}
            cfg = {"output_dir": str(out_dir), "disk_free_gb_threshold": 0.0, "stale_log_minutes": 9999}
            with mock.patch.object(tg, "run_command", return_value=(0, NVIDIA_SMI_3GPU, "")):
                sample = tg.collect_monitor_sample(cfg, state)
            self.assertIn("checkpoint-10", sample["checkpoints"])
            self.assertTrue(sample["trainer_state"]["ok"])

    def test_monitor_once_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with mock.patch.object(tg, "run_command", return_value=(0, NVIDIA_SMI_3GPU, "")):
                code = tg.main(["monitor", "--once", "--output-dir", str(out_dir), "--disk-free-gb-threshold", "0"])
            self.assertIn(code, (tg.EXIT_OK, tg.EXIT_WARN, tg.EXIT_FAIL))
            self.assertTrue((out_dir / "monitor.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
