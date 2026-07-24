from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.api import ReliabilitySession
from train_guard.run.lifecycle import lifecycle_path, summarize_lifecycle


class ApiLifecycleTests(unittest.TestCase):
    def test_session_records_checkpoint_and_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with ReliabilitySession(run_id="api-run", state_dir=root) as session:
                session.observe({"step": 3, "loss": 1.0}, timestamp=1.0)
                session.checkpoint("../checkpoint-3", step=3)
            summary = summarize_lifecycle(lifecycle_path(root))
            self.assertEqual(summary["phase"], "finished")
            self.assertEqual(summary["global_step"], 3)
            self.assertEqual(summary["checkpoints"], ["checkpoint-3"])

    def test_context_exception_records_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "failed"):
                with ReliabilitySession(run_id="api-run", state_dir=root):
                    raise ValueError("failed")
            self.assertEqual(
                summarize_lifecycle(lifecycle_path(root))["phase"],
                "aborted",
            )


if __name__ == "__main__":
    unittest.main()
