from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.domain import Diagnostic, Event, EventKind, Severity


class DomainTests(unittest.TestCase):
    def test_event_round_trip_and_models(self) -> None:
        event = Event("run-1", EventKind.NAN_INF, Severity.CRITICAL, "bad loss", step=3)
        restored = Event.from_dict(event.to_dict())
        self.assertEqual(restored, event)
        self.assertEqual(Diagnostic(event.event_id, "summary").schema_version, "1.0")


if __name__ == "__main__":
    unittest.main()
