"""SQLite cache for media file checks (path + size + mtime)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional

from .. import __version__


class FileCheckCache:
    """Cache verify/hash results keyed by path, size, mtime, kind, tool version."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(db_path))) as conn:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS file_checks (
                        norm_path TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        check_kind TEXT NOT NULL,
                        tool_version TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        PRIMARY KEY (norm_path, size, mtime_ns, check_kind, tool_version)
                    )
                    """
                )

    def get(self, path: Path, check_kind: str) -> Optional[Dict[str, Any]]:
        """Return cached result dict or None."""
        try:
            st = path.stat()
        except OSError:
            return None
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            row = conn.execute(
                """
                SELECT result_json FROM file_checks
                WHERE norm_path=? AND size=? AND mtime_ns=? AND check_kind=? AND tool_version=?
                """,
                (
                    str(path.resolve()),
                    int(st.st_size),
                    int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                    check_kind,
                    __version__,
                ),
            ).fetchone()
        if not row:
            return None
        import json

        try:
            data = json.loads(row[0])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def set(self, path: Path, check_kind: str, result: Dict[str, Any]) -> None:
        """Store result for path metadata."""
        import json

        try:
            st = path.stat()
        except OSError:
            return
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_checks
                    (norm_path, size, mtime_ns, check_kind, tool_version, result_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(path.resolve()),
                        int(st.st_size),
                        int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                        check_kind,
                        __version__,
                        json.dumps(result, ensure_ascii=False),
                    ),
                )
