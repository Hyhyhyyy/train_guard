from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_guard.core.io_util import append_jsonl


class JsonlRotationTests(unittest.TestCase):
    def test_append_uses_lock_and_rotates_at_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            append_jsonl(path, {"value": "first"}, max_bytes=25)
            append_jsonl(path, {"value": "second"}, max_bytes=25)

            previous = path.with_name("events.jsonl.1")
            self.assertEqual(json.loads(previous.read_text(encoding="utf-8")), {"value": "first"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": "second"})
            self.assertTrue(path.with_name(".events.jsonl.lock").exists())


if __name__ == "__main__":
    unittest.main()
