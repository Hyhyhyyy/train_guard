from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from train_guard.domain import Event
from train_guard.reliability import ReliabilityEngine
from train_guard.rules import RuleConfig, RuleEngine
from train_guard.state import AuditLog, StateStore


class _Sink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class ReliabilityPipelineTests(unittest.TestCase):
    def test_persists_rule_state_deduplicates_and_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sink = _Sink()
            store = StateStore(root / "state.sqlite")
            engine = ReliabilityEngine(
                store,
                rules=RuleEngine(RuleConfig(loss_spike_ratio=2.0)),
                sinks=(sink,),
                audit_log=AuditLog(root / "audit.jsonl"),
                notification_every=2,
            )
            self.assertFalse(engine.evaluate("run-1", {"step": 1, "loss": 1.0}, now=1).events)
            first = engine.evaluate("run-1", {"step": 2, "loss": 3.0}, now=2)
            self.assertEqual(len(first.events), 1)
            self.assertEqual(len(sink.events), 1)
            second = engine.evaluate("run-1", {"step": 3, "loss": 9.0}, now=3)
            self.assertEqual(len(second.events), 1)
            self.assertEqual(len(sink.events), 2)
            resolved = engine.evaluate("run-1", {"step": 4, "loss": 10.0}, now=4)
            self.assertEqual(len(resolved.resolved), 1)
            store.close()

            reopened = StateStore(root / "state.sqlite")
            resumed = ReliabilityEngine(
                reopened,
                rules=RuleEngine(RuleConfig(loss_spike_ratio=2.0)),
            )
            event = resumed.evaluate("run-1", {"step": 5, "loss": 25.0}, now=5)
            self.assertEqual(len(event.events), 1)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
