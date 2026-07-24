from __future__ import annotations

import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.domain import Event, EventKind, Severity
from train_guard.reliability import ReliabilityEngine
from train_guard.sinks import WebhookSink
from train_guard.state import AuditLog, StateStore


class _BrokenSink:
    def emit(self, event: Event) -> None:
        del event
        raise OSError("offline")


class NotificationResilienceTests(unittest.TestCase):
    def test_webhook_retries_with_backoff(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.status = 204
        with (
            mock.patch(
                "train_guard.sinks.urllib.request.urlopen",
                side_effect=[urllib.error.URLError("offline"), response],
            ) as urlopen,
            mock.patch("train_guard.sinks.time.sleep") as sleep,
        ):
            WebhookSink(
                "https://example.invalid/hook",
                retries=1,
                backoff_seconds=0.1,
            ).emit(Event("r", EventKind.DISK_LOW, Severity.ERROR, "low"))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_sink_failure_does_not_stop_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = AuditLog(root / "audit.jsonl")
            with StateStore(root / "state.sqlite") as store:
                engine = ReliabilityEngine(
                    store,
                    sinks=[_BrokenSink()],
                    audit_log=audit,
                )
                result = engine.evaluate(
                    "r",
                    {"process_alive": False},
                    now=1.0,
                )
            self.assertEqual(result.events[0].kind, EventKind.PROCESS_DEAD)
            self.assertEqual(result.notified, ())
            self.assertIn("sink_error", {item["type"] for item in audit.records()})


if __name__ == "__main__":
    unittest.main()
