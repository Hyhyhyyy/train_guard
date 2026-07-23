#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""doctor 子命令单元测试。"""

from __future__ import annotations

import json
import sys
import tempfile
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


class TestDoctor(unittest.TestCase):
    """环境自检相关测试。"""

    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            tg.main(["doctor", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_three_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "qwen2_vl", "architectures": ["Qwen2VL"]}),
                encoding="utf-8",
            )
            (model / "model-00001-of-00002.safetensors").write_bytes(b"x")
            (model / "model-00002-of-00002.safetensors").write_bytes(b"y")
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "doctor.json"

            def fake_run(args, timeout=30.0):  # noqa: ANN001
                if args and args[0] == "nvidia-smi":
                    return 0, NVIDIA_SMI_3GPU, ""
                return tg.run_command.__wrapped__(args, timeout) if hasattr(tg.run_command, "__wrapped__") else (1, "", "skip")

            with mock.patch.object(tg, "run_command", side_effect=lambda args, timeout=30.0: (
                (0, NVIDIA_SMI_3GPU, "") if args and args[0] == "nvidia-smi" else (127, "", "not mocked")
            )):
                code = tg.main(
                    [
                        "doctor",
                        "--expected-gpus",
                        "3",
                        "--model-path",
                        str(model),
                        "--json-output",
                        str(out),
                    ]
                )
            self.assertIn(code, (tg.EXIT_OK, tg.EXIT_WARN))
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["command"], "doctor")
            names = {c["name"]: c for c in report["checks"]}
            self.assertEqual(names["gpus"]["details"]["count"], 3)
            self.assertEqual(names["safetensors_index"]["status"], "PASS")

    def test_no_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "doctor.json"
            with mock.patch.object(
                tg,
                "run_command",
                return_value=(127, "", "命令不存在: nvidia-smi"),
            ):
                code = tg.main(["doctor", "--json-output", str(out)])
            self.assertEqual(code, tg.EXIT_FAIL)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["overall_status"], "FAIL")

    def test_query_nvidia_smi_parse(self) -> None:
        with mock.patch.object(tg, "run_command", return_value=(0, NVIDIA_SMI_3GPU, "")):
            info = tg.query_nvidia_smi()
        self.assertTrue(info["ok"])
        self.assertEqual(info["count"], 3)
        self.assertEqual(info["driver_version"], "525.60.13")


if __name__ == "__main__":
    unittest.main()
