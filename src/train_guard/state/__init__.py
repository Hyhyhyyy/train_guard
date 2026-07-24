"""SQLite runtime state and append-only JSONL audit storage."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar

from train_guard.core.io_util import append_jsonl
from train_guard.domain import Event, json_safe, utc_now

SCHEMA_VERSION = 2
_SQLITE_RETRIES = 8
_T = TypeVar("_T")


class StateStore:
    """Transactional local state store safe for use across process restarts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path), timeout=30.0, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        try:
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._retry(lambda: self._connection.execute("PRAGMA journal_mode=WAL").fetchone())
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._initialize()
        except Exception:
            self._connection.close()
            raise

    def _retry(self, operation: Callable[[], _T]) -> _T:
        delay = 0.01
        for attempt in range(_SQLITE_RETRIES):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == _SQLITE_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.25)
        raise AssertionError("unreachable")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._retry(lambda: self._connection.execute("BEGIN IMMEDIATE"))
            try:
                yield
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _initialize(self) -> None:
        statements = (
            """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS run_state (
                    run_id TEXT NOT NULL, key TEXT NOT NULL, value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY (run_id, key)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS offsets (
                    run_id TEXT NOT NULL, source TEXT NOT NULL, offset_value INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY (run_id, source)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS counters (
                    run_id TEXT NOT NULL, name TEXT NOT NULL, value INTEGER NOT NULL,
                    PRIMARY KEY (run_id, name)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS alerts (
                    run_id TEXT NOT NULL, fingerprint TEXT NOT NULL, event_json TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL, opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, resolved_at TEXT,
                    PRIMARY KEY (run_id, fingerprint)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS samples (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    sample_json TEXT NOT NULL
                )
            """,
            """
                CREATE INDEX IF NOT EXISTS samples_run_sequence
                ON samples(run_id, sequence DESC)
            """,
            """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, name)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS recoveries (
                    recovery_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS managed_processes (
                    run_id TEXT PRIMARY KEY,
                    pid INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS control_commands (
                    command_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    outcome_json TEXT,
                    updated_at TEXT NOT NULL
                )
            """,
            """
                CREATE INDEX IF NOT EXISTS control_run_status
                ON control_commands(run_id, status, created_at)
            """,
        )
        with self._transaction():
            for statement in statements:
                self._connection.execute(statement)
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError("state database was created by a newer Train Guard version")
            elif int(row["value"]) < SCHEMA_VERSION:
                self._connection.execute(
                    "UPDATE metadata SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION),),
                )

    def set_run_state(self, run_id: str, key: str, value: Any) -> None:
        encoded = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._transaction():
            self._connection.execute(
                """INSERT INTO run_state VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id, key) DO UPDATE SET value_json=excluded.value_json,
                   updated_at=excluded.updated_at""",
                (run_id, key, encoded, utc_now()),
            )

    def get_run_state(self, run_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._retry(
                lambda: self._connection.execute(
                    "SELECT value_json FROM run_state WHERE run_id=? AND key=?", (run_id, key)
                ).fetchone()
            )
        return default if row is None else json.loads(str(row["value_json"]))

    def set_offset(self, run_id: str, source: str, offset: int) -> None:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._transaction():
            self._connection.execute(
                """INSERT INTO offsets VALUES (?, ?, ?, ?)
                   ON CONFLICT(run_id, source) DO UPDATE SET offset_value=excluded.offset_value,
                   updated_at=excluded.updated_at""",
                (run_id, source, offset, utc_now()),
            )

    def get_offset(self, run_id: str, source: str) -> int:
        with self._lock:
            row = self._retry(
                lambda: self._connection.execute(
                    "SELECT offset_value FROM offsets WHERE run_id=? AND source=?", (run_id, source)
                ).fetchone()
            )
        return 0 if row is None else int(row["offset_value"])

    def increment(self, run_id: str, name: str, amount: int = 1) -> int:
        with self._transaction():
            self._connection.execute(
                """INSERT INTO counters VALUES (?, ?, ?)
                   ON CONFLICT(run_id, name) DO UPDATE SET value=value+excluded.value""",
                (run_id, name, amount),
            )
            row = self._connection.execute(
                "SELECT value FROM counters WHERE run_id=? AND name=?", (run_id, name)
            ).fetchone()
        return int(row["value"])

    def record_alert(self, fingerprint: str, event: Event) -> int:
        """Open/reopen or increment an alert and return occurrence count."""
        occurrence, _ = self.record_alert_transition(fingerprint, event)
        return occurrence

    def record_alert_transition(self, fingerprint: str, event: Event) -> tuple[int, bool]:
        """Record an alert and return ``(occurrence, reopened)`` atomically."""
        now = utc_now()
        with self._transaction():
            previous = self._connection.execute(
                "SELECT resolved_at FROM alerts WHERE run_id=? AND fingerprint=?",
                (event.run_id, fingerprint),
            ).fetchone()
            reopened = previous is not None and previous["resolved_at"] is not None
            self._connection.execute(
                """INSERT INTO alerts VALUES (?, ?, ?, 1, ?, ?, NULL)
                   ON CONFLICT(run_id, fingerprint) DO UPDATE SET
                   event_json=excluded.event_json,
                   occurrence_count=CASE WHEN alerts.resolved_at IS NOT NULL
                       THEN 1 ELSE alerts.occurrence_count+1 END,
                   opened_at=CASE WHEN alerts.resolved_at IS NOT NULL
                       THEN excluded.opened_at ELSE alerts.opened_at END,
                   updated_at=excluded.updated_at,
                   resolved_at=NULL""",
                (event.run_id, fingerprint, event.to_json(), now, now),
            )
            row = self._connection.execute(
                "SELECT occurrence_count FROM alerts WHERE run_id=? AND fingerprint=?",
                (event.run_id, fingerprint),
            ).fetchone()
        return int(row["occurrence_count"]), reopened

    def resolve_alert(self, run_id: str, fingerprint: str) -> bool:
        with self._transaction():
            cursor = self._connection.execute(
                "UPDATE alerts SET resolved_at=?, updated_at=? WHERE run_id=? AND fingerprint=? AND resolved_at IS NULL",
                (utc_now(), utc_now(), run_id, fingerprint),
            )
        return cursor.rowcount > 0

    def active_alerts(self, run_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        query = "SELECT * FROM alerts WHERE resolved_at IS NULL"
        parameters: tuple[str, ...] = ()
        if run_id is not None:
            query += " AND run_id=?"
            parameters = (run_id,)
        query += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._retry(lambda: self._connection.execute(query, parameters).fetchall())
        for row in rows:
            item = dict(row)
            item["event"] = json.loads(item.pop("event_json"))
            yield item

    def list_runs(self) -> tuple[str, ...]:
        query = """
            SELECT run_id FROM run_state
            UNION SELECT run_id FROM alerts
            UNION SELECT run_id FROM samples
            UNION SELECT run_id FROM managed_processes
            ORDER BY run_id
        """
        with self._lock:
            rows = self._retry(lambda: self._connection.execute(query).fetchall())
        return tuple(str(row["run_id"]) for row in rows)

    def record_sample(
        self,
        run_id: str,
        observed_at: float,
        sample: Mapping[str, Any],
        *,
        retention: int = 2000,
    ) -> None:
        if retention < 1:
            raise ValueError("sample retention must be positive")
        normalized = dict(sample)
        if "metrics" not in normalized:
            normalized = {"timestamp": observed_at, "metrics": normalized}
        encoded = json.dumps(
            json_safe(normalized),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction():
            self._connection.execute(
                "INSERT INTO samples (run_id, observed_at, sample_json) VALUES (?, ?, ?)",
                (run_id, observed_at, encoded),
            )
            self._connection.execute(
                """DELETE FROM samples WHERE run_id=? AND sequence NOT IN (
                       SELECT sequence FROM samples WHERE run_id=?
                       ORDER BY sequence DESC LIMIT ?
                   )""",
                (run_id, run_id, retention),
            )

    def latest_sample(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._retry(
                lambda: self._connection.execute(
                    """SELECT observed_at, sample_json FROM samples
                       WHERE run_id=? ORDER BY sequence DESC LIMIT 1""",
                    (run_id,),
                ).fetchone()
            )
        if row is None:
            return {}
        sample = dict(json.loads(str(row["sample_json"])))
        sample.setdefault("timestamp", float(row["observed_at"]))
        return sample

    def metric_series(
        self,
        run_id: str,
        *,
        limit: int = 120,
    ) -> Dict[str, tuple[float, ...]]:
        if limit < 1 or limit > 5000:
            raise ValueError("metric series limit must be between 1 and 5000")
        with self._lock:
            rows = self._retry(
                lambda: self._connection.execute(
                    """SELECT sample_json FROM samples WHERE run_id=?
                       ORDER BY sequence DESC LIMIT ?""",
                    (run_id, limit),
                ).fetchall()
            )
        series: Dict[str, list[float]] = {}
        for row in reversed(rows):
            sample = json.loads(str(row["sample_json"]))
            metrics = sample.get("metrics") if isinstance(sample, dict) else None
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                series.setdefault(str(key), []).append(float(value))
        return {key: tuple(values) for key, values in series.items()}

    def record_checkpoint(
        self,
        run_id: str,
        name: str,
        status: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        encoded = json.dumps(
            json_safe(dict(details or {})),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction():
            self._connection.execute(
                """INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, name) DO UPDATE SET
                   status=excluded.status, details_json=excluded.details_json,
                   updated_at=excluded.updated_at""",
                (run_id, name, status, encoded, utc_now()),
            )

    def checkpoint_history(self, run_id: str) -> Iterator[Dict[str, Any]]:
        with self._lock:
            rows = self._retry(
                lambda: self._connection.execute(
                    """SELECT * FROM checkpoints WHERE run_id=?
                       ORDER BY updated_at DESC""",
                    (run_id,),
                ).fetchall()
            )
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            yield item

    def record_recovery(
        self,
        run_id: str,
        action: str,
        status: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        encoded = json.dumps(
            json_safe(dict(details or {})),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction():
            self._connection.execute(
                """INSERT INTO recoveries
                   (run_id, action, status, details_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, action, status, encoded, utc_now()),
            )

    def recovery_history(self, run_id: str, limit: int = 100) -> Iterator[Dict[str, Any]]:
        with self._lock:
            rows = self._retry(
                lambda: self._connection.execute(
                    """SELECT * FROM recoveries WHERE run_id=?
                       ORDER BY recovery_id DESC LIMIT ?""",
                    (run_id, limit),
                ).fetchall()
            )
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            yield item

    def register_managed_process(
        self,
        run_id: str,
        pid: int,
        status: str,
        capabilities: tuple[str, ...],
    ) -> None:
        encoded = json.dumps(list(capabilities), allow_nan=False)
        with self._transaction():
            self._connection.execute(
                """INSERT INTO managed_processes VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET pid=excluded.pid,
                   status=excluded.status, capabilities_json=excluded.capabilities_json,
                   updated_at=excluded.updated_at""",
                (run_id, pid, status, encoded, utc_now()),
            )

    def managed_process(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._retry(
                lambda: self._connection.execute(
                    "SELECT * FROM managed_processes WHERE run_id=?",
                    (run_id,),
                ).fetchone()
            )
        if row is None:
            return {}
        item = dict(row)
        item["capabilities"] = json.loads(item.pop("capabilities_json"))
        return item

    def enqueue_control(self, request: object) -> bool:
        to_dict = getattr(request, "to_dict", None)
        if to_dict is None:
            raise TypeError("control request must provide to_dict")
        payload = dict(to_dict())
        encoded = json.dumps(
            json_safe(dict(payload.get("parameters") or {})),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction():
            process = self._connection.execute(
                "SELECT capabilities_json FROM managed_processes WHERE run_id=?",
                (str(payload["run_id"]),),
            ).fetchone()
            if process is None:
                raise ValueError("control is limited to managed processes")
            capabilities = json.loads(str(process["capabilities_json"]))
            if str(payload["action"]) not in capabilities:
                raise ValueError("control action is unavailable for this managed process")
            cursor = self._connection.execute(
                """INSERT OR IGNORE INTO control_commands
                   (command_id, run_id, action, parameters_json, created_at,
                    expires_at, status, outcome_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)""",
                (
                    str(payload["command_id"]),
                    str(payload["run_id"]),
                    str(payload["action"]),
                    encoded,
                    float(payload["created_at"]),
                    float(payload["expires_at"]),
                    utc_now(),
                ),
            )
        return cursor.rowcount > 0

    def claim_control(self, run_id: str, now: float) -> Optional[Dict[str, Any]]:
        with self._transaction():
            self._connection.execute(
                """UPDATE control_commands SET status='expired', updated_at=?
                   WHERE run_id=? AND status='queued' AND expires_at<=?""",
                (utc_now(), run_id, now),
            )
            row = self._connection.execute(
                """SELECT * FROM control_commands WHERE run_id=? AND status='queued'
                   ORDER BY created_at LIMIT 1""",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            self._connection.execute(
                """UPDATE control_commands SET status='running', updated_at=?
                   WHERE command_id=?""",
                (utc_now(), row["command_id"]),
            )
        item = dict(row)
        item["parameters"] = json.loads(item.pop("parameters_json"))
        return item

    def complete_control(
        self,
        command_id: str,
        status: str,
        outcome: Mapping[str, Any],
    ) -> None:
        if status not in {"succeeded", "failed", "denied"}:
            raise ValueError("invalid control outcome status")
        encoded = json.dumps(
            json_safe(dict(outcome)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._transaction():
            self._connection.execute(
                """UPDATE control_commands SET status=?, outcome_json=?, updated_at=?
                   WHERE command_id=?""",
                (status, encoded, utc_now(), command_id),
            )

    def reset_run(self, run_id: str) -> None:
        """Remove persisted watcher/rule state for a newly detected lifecycle."""
        with self._transaction():
            for table in (
                "run_state",
                "offsets",
                "counters",
                "alerts",
                "samples",
                "checkpoints",
                "recoveries",
                "managed_processes",
                "control_commands",
            ):
                self._connection.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AuditLog:
    """Append-only JSONL audit trail."""

    def __init__(self, path: Path, max_bytes: Optional[int] = 10 * 1024**2) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        entry = dict(json_safe({"timestamp": utc_now(), **dict(record)}))
        json.dumps(entry, ensure_ascii=False, allow_nan=False)
        with self._lock:
            append_jsonl(self.path, entry, max_bytes=self.max_bytes)

    def records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield dict(json.loads(line))


__all__ = ["AuditLog", "SCHEMA_VERSION", "StateStore"]
