from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.domain import Event, EventKind, Severity
from train_guard.sinks import ConsoleSink, JsonlSink, OtelJsonSink, prometheus_text


class SinkTests(unittest.TestCase):
    def test_public_api_and_framework_exports_are_lazy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(root / "src"), environment.get("PYTHONPATH", "")]
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, train_guard; "
                    "assert train_guard.__version__ == '0.6.0rc1'; "
                    "assert 'train_guard.api' not in sys.modules; "
                    "assert train_guard.check_run is not None; "
                    "assert 'train_guard.api' in sys.modules; "
                    "import train_guard.frameworks as frameworks; "
                    "assert 'train_guard.frameworks.huggingface' not in sys.modules; "
                    "assert frameworks.TrainGuardCallback is not None"
                ),
            ],
            cwd=root.parent,
            env=environment,
            check=True,
        )

    def test_local_exports(self) -> None:
        event = Event("r", EventKind.DISK_LOW, Severity.ERROR, "low")
        output = io.StringIO()
        ConsoleSink(output).emit(event)
        self.assertIn("disk_low", output.getvalue())
        self.assertIn('kind="disk_low"', prometheus_text([event]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            JsonlSink(root / "events.jsonl").emit(event)
            OtelJsonSink(root / "otel.jsonl").emit(event)
            self.assertEqual(
                json.loads((root / "events.jsonl").read_text())["event_id"], event.event_id
            )
            self.assertEqual(json.loads((root / "otel.jsonl").read_text())["severityText"], "ERROR")


if __name__ == "__main__":
    unittest.main()
