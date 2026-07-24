from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.supervisor import (
    FileCheckpointValidator,
    ProcessSpec,
    RecoveryGuard,
    RecoveryPolicy,
    supervise,
)


class SupervisorTests(unittest.TestCase):
    def test_checkpoint_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "weights.bin").write_bytes(b"weights")
            validator = FileCheckpointValidator(["weights.bin", "state.json"])
            self.assertFalse(validator.validate(checkpoint).valid)
            (checkpoint / "state.json").write_text("{}", encoding="utf-8")
            self.assertTrue(validator.validate(checkpoint).valid)

    def test_restart_loop_is_bounded(self) -> None:
        guard = RecoveryGuard(RecoveryPolicy(max_restarts=2, window_seconds=100))
        self.assertTrue(guard.permit_restart(0))
        self.assertTrue(guard.permit_restart(1))
        self.assertFalse(guard.permit_restart(2))
        self.assertTrue(guard.permit_restart(101))

    def test_supervise_restarts_only_with_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "state.json").write_text("{}", encoding="utf-8")
            marker = root / "attempted"
            code = (
                "from pathlib import Path; import sys; "
                f"p=Path({str(marker)!r}); "
                "exists=p.exists(); p.write_text('1'); sys.exit(0 if exists else 7)"
            )
            result = supervise(
                ProcessSpec(sys.executable, ("-c", code)),
                restart_enabled=True,
                recovery_guard=RecoveryGuard(RecoveryPolicy(max_restarts=1, window_seconds=60)),
                checkpoint_path=checkpoint,
                checkpoint_validator=FileCheckpointValidator(["state.json"]),
            )
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.restart_count, 1)


if __name__ == "__main__":
    unittest.main()
