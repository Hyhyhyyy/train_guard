from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.domain import Event, EventKind, Severity
from train_guard.sinks import PrometheusFileSink


class ExportSinkTests(unittest.TestCase):
    def test_prometheus_textfile_accumulates_event_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_guard.prom"
            sink = PrometheusFileSink(path)
            event = Event("r", EventKind.DISK_LOW, Severity.ERROR, "low")
            sink.emit(event)
            sink.emit(event)
            text = path.read_text(encoding="utf-8")
            self.assertIn('kind="disk_low",severity="error"} 2', text)


if __name__ == "__main__":
    unittest.main()
