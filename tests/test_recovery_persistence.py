from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.supervisor import RecoveryGuard, RecoveryPolicy


class RecoveryPersistenceTests(unittest.TestCase):
    def test_restored_budget_is_updated_immediately(self) -> None:
        snapshots: list[tuple[float, ...]] = []
        guard = RecoveryGuard(
            RecoveryPolicy(max_restarts=2, window_seconds=100),
            restart_times=[90],
            on_change=lambda values: snapshots.append(tuple(values)),
        )
        self.assertTrue(guard.permit_restart(100))
        self.assertEqual(snapshots, [(90.0, 100)])
        self.assertFalse(guard.permit_restart(101))


if __name__ == "__main__":
    unittest.main()
