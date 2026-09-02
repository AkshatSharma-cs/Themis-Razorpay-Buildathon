"""
audit_db.py — Append-only, hash-chained audit log for Sentinel.

WHY THIS EXISTS:
Judges and regulators don't need to trust our word that the audit log wasn't
edited after the fact — they can verify it. Each row's hash commits to the
previous row's hash + this row's content, so changing any historical row
(or deleting one) breaks every hash after it. verify_chain() walks the
whole table and proves integrity in O(n), no external service required.

This is intentionally NOT a general-purpose event store. It is a narrow,
tamper-evident ledger for exactly the events that matter for this track:
score computed, decision issued, cooling-off issued, cooling-off released,
daily-cap reached. Keep it that way — don't dump debug logs in here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from backend.narration import get_narration

DB_PATH = os.environ.get("SENTINEL_AUDIT_DB", "sentinel_audit.sqlite3")

# Single lock because SQLite serializes writers anyway; keeps hash-chain
# appends atomic without reaching for a heavier concurrency model.
_write_lock = threading.Lock()

GENESIS_HASH = "0" * 64  # hash of the (nonexistent) row before the first row


# --------------------------------------------------------------------------- #
# Connection helpers
# --------------------------------------------------------------------------- #

@contextmanager
def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")  # avoid corrupt-on-crash surprises
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    """Create the audit table if it doesn't exist. Safe to call on every boot."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id      TEXT NOT NULL UNIQUE,
                tx_id         TEXT NOT NULL,
                payer_vpa     TEXT,
                event_type    TEXT NOT NULL,
                payload_json  TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                prev_hash     TEXT NOT NULL,
                row_hash      TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_tx ON audit_log(tx_id);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_payer_day "
            "ON audit_log(payer_vpa, event_type, created_at);"
        )


# --------------------------------------------------------------------------- #
# Canonicalization + hashing
# --------------------------------------------------------------------------- #

def _canonical_json(payload: dict) -> str:
    """Deterministic serialization so the same content always hashes the same."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _row_hash(prev_hash: str, event_id: str, tx_id: str, payer_vpa: Optional[str],
              event_type: str, payload_json: str, created_at: str) -> str:
    blob = "|".join([prev_hash, event_id, tx_id, payer_vpa or "", event_type,
                      payload_json, created_at]).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _get_last_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["row_hash"] if row else GENESIS_HASH


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AuditRow:
    id: int
    event_id: str
    tx_id: str
    payer_vpa: Optional[str]
    event_type: str
    payload: dict
    created_at: str
    prev_hash: str
    row_hash: str


def append_event(
    tx_id: str,
    event_type: str,
    payload: dict,
    payer_vpa: Optional[str] = None,
    db_path: str = DB_PATH,
) -> AuditRow:
    """
    Append one tamper-evident row. This is the ONLY way rows enter the table —
    there is deliberately no update_event()/delete_event() function. If you
    need to "correct" history, append a new event that explains the correction;
    never mutate a past row.
    """
    event_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    payload_json = _canonical_json(payload)

    with _write_lock, _connect(db_path) as conn:
        prev_hash = _get_last_hash(conn)
        row_hash = _row_hash(prev_hash, event_id, tx_id, payer_vpa, event_type,
                              payload_json, created_at)
        conn.execute(
            """
            INSERT INTO audit_log
                (event_id, tx_id, payer_vpa, event_type, payload_json,
                 created_at, prev_hash, row_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, tx_id, payer_vpa, event_type, payload_json,
             created_at, prev_hash, row_hash),
        )
        new_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    return AuditRow(new_id, event_id, tx_id, payer_vpa, event_type,
                     payload, created_at, prev_hash, row_hash)


def get_audit_trail(tx_id: str, db_path: str = DB_PATH) -> list[dict]:
    """Return the full ordered event history for one transaction."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE tx_id = ? ORDER BY id ASC", (tx_id,)
        ).fetchall()
    return [
        {
            "id": r["id"],
            "event_id": r["event_id"],
            "tx_id": r["tx_id"],
            "payer_vpa": r["payer_vpa"],
            "event_type": r["event_type"],
            "payload": json.loads(r["payload_json"]),
            "created_at": r["created_at"],
            "prev_hash": r["prev_hash"],
            "row_hash": r["row_hash"],
        }
        for r in rows
    ]


def count_events_today(
    payer_vpa: str, event_type: str, db_path: str = DB_PATH
) -> int:
    """Used for the daily cooling-off cap. UTC calendar day, matching created_at."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM audit_log
            WHERE payer_vpa = ? AND event_type = ?
              AND substr(created_at, 1, 10) = ?
            """,
            (payer_vpa, event_type, today),
        ).fetchone()
    return int(row["n"])


def verify_chain(db_path: str = DB_PATH) -> dict:
    """
    Walk the entire table and recompute every row's hash from scratch.
    Returns a report suitable for a demo screenshot:
      { "ok": bool, "rows_checked": int, "first_broken_row_id": int | None,
        "reason": str | None }

    This is the concrete, demoable differentiator: run this against an
    untouched DB (ok=True), then hand-edit one payload_json with a raw
    sqlite3 UPDATE and re-run it live (ok=False, points at the exact row).
    """
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()

    expected_prev = GENESIS_HASH
    for r in rows:
        if r["prev_hash"] != expected_prev:
            return {
                "ok": False,
                "rows_checked": r["id"],
                "first_broken_row_id": r["id"],
                "reason": (
                    f"row {r['id']}: stored prev_hash does not match the actual "
                    f"hash of the previous row (chain link broken)"
                ),
            }
        recomputed = _row_hash(
            r["prev_hash"], r["event_id"], r["tx_id"], r["payer_vpa"],
            r["event_type"], r["payload_json"], r["created_at"],
        )
        if recomputed != r["row_hash"]:
            return {
                "ok": False,
                "rows_checked": r["id"],
                "first_broken_row_id": r["id"],
                "reason": (
                    f"row {r['id']}: stored row_hash does not match a hash "
                    f"recomputed from its own content (row was edited in place)"
                ),
            }
        expected_prev = r["row_hash"]

    return {
        "ok": True,
        "rows_checked": len(rows),
        "first_broken_row_id": None,
        "reason": None,
    }


if __name__ == "__main__":
    # Quick manual smoke test: python audit_db.py
    init_db()
    a = append_event("txn_demo_1", "SCORE_COMPUTED", {"probability": 0.82}, "alice@upi")
    b = append_event("txn_demo_1", "COOLING_OFF_ISSUED",
                      {"duration_hours": 2}, "alice@upi")
    print("trail:", get_audit_trail("txn_demo_1"))
    print("verify (should be ok=True):", verify_chain())

    # Tamper with a row directly and re-verify to show detection.
    with _connect() as conn:
        conn.execute(
            "UPDATE audit_log SET payload_json = ? WHERE id = ?",
            (_canonical_json({"probability": 0.01}), a.id),
        )
    print("verify after tamper (should be ok=False):", verify_chain())
