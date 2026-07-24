from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.domain import EventKind
from train_guard.rules import RuleConfig, RuleContext, RuleEngine


class RuleTests(unittest.TestCase):
    def test_all_initial_rule_families(self) -> None:
        engine = RuleEngine(RuleConfig(stall_seconds=10, checkpoint_stale_seconds=10))
        engine.evaluate(
            RuleContext("r", {"step": 1, "loss": 1.0, "grad_norm": 1.0, "throughput": 10.0}, 0)
        )
        values = {
            "step": 1,
            "loss": float("nan"),
            "grad_norm": 10.0,
            "throughput": 1.0,
            "gpu_util_percent": 0,
            "gpu_temperature_c": 99,
            "disk_free_bytes": 1,
            "process_alive": False,
            "cuda_oom": "CUDA out of memory",
            "nccl_error": "NCCL ERROR",
            "gpu_xid": "NVRM: Xid 79",
            "checkpoint_age_seconds": 99,
            "checkpoint_valid": False,
        }
        kinds = {event.kind for event in engine.evaluate(RuleContext("r", values, 20))}
        expected = {
            EventKind.NAN_INF,
            EventKind.GRAD_SPIKE,
            EventKind.STEP_STALLED,
            EventKind.THROUGHPUT_DROP,
            EventKind.GPU_IDLE,
            EventKind.GPU_OVERHEAT,
            EventKind.DISK_LOW,
            EventKind.PROCESS_DEAD,
            EventKind.CUDA_OOM,
            EventKind.NCCL_ERROR,
            EventKind.GPU_XID,
            EventKind.CHECKPOINT_STALE,
            EventKind.CHECKPOINT_CORRUPT,
        }
        self.assertTrue(expected.issubset(kinds))

    def test_loss_spike(self) -> None:
        engine = RuleEngine()
        engine.evaluate(RuleContext("r", {"loss": 1.0}, 0))
        events = engine.evaluate(RuleContext("r", {"loss": 4.0}, 1))
        self.assertIn(EventKind.LOSS_SPIKE, {event.kind for event in events})


if __name__ == "__main__":
    unittest.main()
