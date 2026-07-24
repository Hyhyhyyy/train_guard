from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.dashboard import create_server
from train_guard.domain import Event, EventKind, Severity
from train_guard.state import StateStore


class DashboardTests(unittest.TestCase):
    def test_local_server_reads_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with StateStore(Path(temporary) / "state.sqlite") as store:
                store.record_alert("disk", Event("r", EventKind.DISK_LOW, Severity.ERROR, "low"))
                server = create_server(store, port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    port = int(server.server_address[1])
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/alerts", timeout=2
                    ) as response:
                        payload = json.loads(response.read())
                    self.assertEqual(payload[0]["event"]["kind"], "disk_low")
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

    def test_external_bind_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with StateStore(Path(temporary) / "state.sqlite") as store:
                with self.assertRaises(ValueError):
                    create_server(store, host="0.0.0.0")


if __name__ == "__main__":
    unittest.main()
