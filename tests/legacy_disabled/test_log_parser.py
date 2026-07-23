#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志解析专项测试（满足独立测试文件要求）。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_guard as tg  # noqa: E402


class TestLogParserFile(unittest.TestCase):
    """覆盖常见日志格式与边界。"""

    def test_combined_line(self) -> None:
        line = "{'loss': 0.123, 'grad_norm': 2.5, 'learning_rate': 1e-5, 'epoch': 0.5, 'step': 100}"
        m = tg.parse_training_metrics(line)
        self.assertAlmostEqual(m["loss"], 0.123)
        self.assertAlmostEqual(m["grad_norm"], 2.5)
        self.assertAlmostEqual(m["learning_rate"], 1e-5)
        self.assertAlmostEqual(m["epoch"], 0.5)
        self.assertEqual(m["step"], 100.0)

    def test_eval_loss_isolation(self) -> None:
        m = tg.parse_training_metrics("train metrics eval_loss=3.14 loss=2.71")
        self.assertAlmostEqual(m["eval_loss"], 3.14)
        self.assertAlmostEqual(m["loss"], 2.71)

    def test_parse_numeric_token(self) -> None:
        self.assertTrue(math.isnan(tg.parse_numeric_token("NaN")))  # type: ignore[arg-type]
        self.assertTrue(math.isinf(tg.parse_numeric_token("inf")))  # type: ignore[arg-type]
        self.assertEqual(tg.parse_numeric_token("abc"), None)


if __name__ == "__main__":
    unittest.main()
