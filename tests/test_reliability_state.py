from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.domain import Event, EventKind, Severity
from train_guard.state import AuditLog, StateStore


class StateTests(unittest.TestCase):
    def test_state_alert_lifecycle_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with StateStore(root / "state.sqlite") as store:
                store.set_run_state("r", "phase", {"name": "train"})
                store.set_offset("r", "log", 42)
                self.assertEqual(store.get_run_state("r", "phase")["name"], "train")
                self.assertEqual(store.get_offset("r", "log"), 42)
                self.assertEqual(store.increment("r", "events", 2), 2)
                event = Event("r", EventKind.PROCESS_DEAD, Severity.CRITICAL, "dead")
                self.assertEqual(store.record_alert("process", event), 1)
                self.assertEqual(store.record_alert("process", event), 2)
                self.assertEqual(len(list(store.active_alerts("r"))), 1)
                self.assertTrue(store.resolve_alert("r", "process"))
                self.assertEqual(list(store.active_alerts("r")), [])
            audit = AuditLog(root / "audit.jsonl")
            audit.append({"operation": "resolve"})
            self.assertEqual(list(audit.records())[0]["operation"], "resolve")

    def test_rejects_database_from_newer_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "future.sqlite"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute("INSERT INTO metadata VALUES ('schema_version', '999')")
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(RuntimeError, "newer Train Guard"):
                StateStore(path)


if __name__ == "__main__":
    unittest.main()
