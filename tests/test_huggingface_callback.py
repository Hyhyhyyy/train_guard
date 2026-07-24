from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.frameworks.huggingface import TrainGuardCallback
from train_guard.rules import RuleConfig
from train_guard.run.lifecycle import lifecycle_path, summarize_lifecycle


class HuggingFaceCallbackTests(unittest.TestCase):
    def test_observes_logs_without_transformers_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = object()
            callback = TrainGuardCallback(
                run_id="callback-run",
                state_dir=Path(temporary),
                rule_config=RuleConfig(loss_spike_ratio=2.0),
            )
            state = SimpleNamespace(global_step=1)
            self.assertIs(callback.on_train_begin(None, state, control), control)
            self.assertIs(
                callback.on_log(None, state, control, logs={"loss": 1.0}),
                control,
            )
            state.global_step = 2
            callback.on_log(None, state, control, logs={"loss": 3.0})
            self.assertIs(callback.on_save(None, state, control), control)
            self.assertIsNotNone(callback.last_result)
            assert callback.last_result is not None
            self.assertEqual(len(callback.last_result.events), 1)
            self.assertIs(callback.on_train_end(None, state, control), control)
            self.assertTrue((Path(temporary) / "reliability_events.jsonl").is_file())
            summary = summarize_lifecycle(lifecycle_path(Path(temporary)))
            self.assertEqual(summary["phase"], "finished")
            self.assertEqual(summary["checkpoints"], ["checkpoint-2"])


if __name__ == "__main__":
    unittest.main()
