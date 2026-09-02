"""
tests/test_audit_chain.py

Exercises backend/audit_db.py's tamper-evident hash chain directly against
a private temp SQLite file (never the default sentinel_audit.sqlite3), and
confirms OTPs never make it into the permanent, hash-chained log -- only
the release *event* does (per ARCHITECTURE.md section 4 / narration.py's
"What's deliberately NOT in the chain" note).
"""
from __future__ import annotations

import sqlite3

import pytest

import backend.audit_db as audit_db


@pytest.fixture
def temp_db(tmp_path) -> str:
    db_path = str(tmp_path / "audit_test.sqlite3")
    audit_db.init_db(db_path=db_path)
    return db_path


def _seed_rows(db_path: str, n: int = 5):
    rows = []
    for i in range(n):
        row = audit_db.append_event(
            tx_id=f"txn_{i:03d}",
            event_type="SCORE_COMPUTED",
            payload={"probability": 0.1 * i},
            payer_vpa="alice@upi",
            db_path=db_path,
        )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Core hash-chain behavior
# --------------------------------------------------------------------------- #

def test_verify_chain_ok_on_untouched_chain(temp_db):
    _seed_rows(temp_db, n=6)
    report = audit_db.verify_chain(db_path=temp_db)
    assert report["ok"] is True, f"expected an untouched chain to verify clean, got {report}"
    assert report["rows_checked"] == 6, (
        f"expected all 6 written rows to be checked, got rows_checked={report['rows_checked']}"
    )
    assert report["first_broken_row_id"] is None


def test_verify_chain_detects_tampered_row_and_names_it(temp_db):
    rows = _seed_rows(temp_db, n=6)
    tampered_row = rows[2]  # the 3rd row written

    conn = sqlite3.connect(temp_db)
    try:
        conn.execute(
            "UPDATE audit_log SET payload_json = ? WHERE id = ?",
            ('{"probability":0.999}', tampered_row.id),
        )
        conn.commit()
    finally:
        conn.close()

    report = audit_db.verify_chain(db_path=temp_db)
    assert report["ok"] is False, "verify_chain() did not detect a hand-edited row"
    assert report["first_broken_row_id"] == tampered_row.id, (
        f"verify_chain() flagged row id {report['first_broken_row_id']}, but the "
        f"row that was actually tampered with is id {tampered_row.id}"
    )
    assert report["reason"] is not None and "row_hash" in report["reason"], (
        f"expected the failure reason to point at a row_hash mismatch, got: {report['reason']!r}"
    )


def test_verify_chain_detects_tampering_with_an_earlier_row(temp_db):
    """Tampering with an early row should still be pinpointed correctly,
    not misattributed to a later row just because the chain 'breaks'
    downstream too."""
    rows = _seed_rows(temp_db, n=8)
    tampered_row = rows[0]

    conn = sqlite3.connect(temp_db)
    try:
        conn.execute(
            "UPDATE audit_log SET tx_id = ? WHERE id = ?",
            ("txn_TAMPERED", tampered_row.id),
        )
        conn.commit()
    finally:
        conn.close()

    report = audit_db.verify_chain(db_path=temp_db)
    assert report["ok"] is False
    assert report["first_broken_row_id"] == tampered_row.id, (
        f"expected the earliest tampered row (id={tampered_row.id}) to be reported, "
        f"got id={report['first_broken_row_id']}"
    )


def test_append_event_is_the_only_way_rows_enter_the_table():
    """audit_db.py's own docstring says there is deliberately no
    update_event()/delete_event() -- confirm that contract still holds."""
    assert not hasattr(audit_db, "update_event"), (
        "audit_db.update_event exists -- the audit log is documented as "
        "append-only; a mutation function violates that guarantee."
    )
    assert not hasattr(audit_db, "delete_event"), (
        "audit_db.delete_event exists -- the audit log is documented as "
        "append-only; a deletion function violates that guarantee."
    )


def test_get_audit_trail_returns_rows_in_order(temp_db):
    _seed_rows(temp_db, n=4)
    audit_db.append_event(
        tx_id="txn_003", event_type="COOLING_OFF_ISSUED",
        payload={"duration_hours": 2}, payer_vpa="alice@upi", db_path=temp_db,
    )
    trail = audit_db.get_audit_trail("txn_003", db_path=temp_db)
    assert [e["event_type"] for e in trail] == ["SCORE_COMPUTED", "COOLING_OFF_ISSUED"], (
        f"expected the trail for txn_003 in insertion order, got {trail}"
    )


def test_count_events_today_only_counts_matching_payer_and_type(temp_db):
    audit_db.append_event("t1", "COOLING_OFF_ISSUED", {}, "alice@upi", db_path=temp_db)
    audit_db.append_event("t2", "COOLING_OFF_ISSUED", {}, "alice@upi", db_path=temp_db)
    audit_db.append_event("t3", "COOLING_OFF_ISSUED", {}, "bob@upi", db_path=temp_db)
    audit_db.append_event("t4", "SCORE_COMPUTED", {}, "alice@upi", db_path=temp_db)

    n_alice = audit_db.count_events_today("alice@upi", "COOLING_OFF_ISSUED", db_path=temp_db)
    n_bob = audit_db.count_events_today("bob@upi", "COOLING_OFF_ISSUED", db_path=temp_db)

    assert n_alice == 2, f"expected 2 COOLING_OFF_ISSUED events for alice today, got {n_alice}"
    assert n_bob == 1, f"expected 1 COOLING_OFF_ISSUED event for bob today, got {n_bob}"


# --------------------------------------------------------------------------- #
# OTPs must never enter the permanent hash-chained log -- only the release
# EVENT (COOLING_OFF_RELEASED). Exercised through serve.py's real release
# flow, against a temp audit DB.
# --------------------------------------------------------------------------- #

@pytest.fixture
def serve_module(monkeypatch, tmp_path):
    import backend.serve as serve

    temp_db = str(tmp_path / "release_flow_audit.sqlite3")
    audit_db.init_db(db_path=temp_db)

    # Capture the ORIGINAL function before patching -- serve.audit_db is the
    # same module object as this test's `audit_db`, so a lambda that looks up
    # `audit_db.append_event` at call time (after patching) would recurse
    # into itself forever.
    original_append_event = audit_db.append_event
    monkeypatch.setattr(
        serve.audit_db, "append_event",
        lambda *a, **kw: original_append_event(*a, **{**kw, "db_path": temp_db}),
    )
    serve._otp_store.clear()
    return serve, temp_db


def test_otp_never_appears_in_the_hash_chained_audit_log(serve_module):
    serve, temp_db = serve_module
    from backend.serve import ReleaseRequestOtp, ReleaseConfirm

    tx_id = "txn_release_test"
    payer = "dave@upi"

    otp_response = serve.request_otp(ReleaseRequestOtp(tx_id=tx_id, payer_vpa=payer))
    real_otp = otp_response["otp_demo_only"]

    confirm_result = serve.confirm_release(ReleaseConfirm(
        tx_id=tx_id, payer_vpa=payer, otp_code=real_otp, confirm_intent=True,
    ))
    assert confirm_result["released"] is True

    trail = audit_db.get_audit_trail(tx_id, db_path=temp_db)
    event_types = [e["event_type"] for e in trail]
    assert "COOLING_OFF_RELEASED" in event_types, (
        f"expected a COOLING_OFF_RELEASED event in the audit trail, got {event_types}"
    )

    # The OTP secret itself must never be embedded anywhere in the
    # permanent, hash-chained rows for this transaction -- only the fact
    # that a valid release happened.
    for event in trail:
        payload_str = str(event["payload"])
        assert real_otp not in payload_str, (
            f"found the raw OTP {real_otp!r} inside a hash-chained audit row "
            f"(event_type={event['event_type']}) -- OTPs must stay ephemeral, "
            f"never written to the permanent log"
        )


def test_failed_release_attempts_are_logged_without_the_otp(serve_module):
    serve, temp_db = serve_module
    from backend.serve import ReleaseRequestOtp, ReleaseConfirm
    from fastapi import HTTPException

    tx_id = "txn_release_fail_test"
    payer = "erin@upi"
    otp_response = serve.request_otp(ReleaseRequestOtp(tx_id=tx_id, payer_vpa=payer))
    real_otp = otp_response["otp_demo_only"]

    wrong_otp = "000000" if real_otp != "000000" else "111111"
    with pytest.raises(HTTPException):
        serve.confirm_release(ReleaseConfirm(
            tx_id=tx_id, payer_vpa=payer, otp_code=wrong_otp, confirm_intent=True,
        ))

    trail = audit_db.get_audit_trail(tx_id, db_path=temp_db)
    event_types = [e["event_type"] for e in trail]
    assert "RELEASE_ATTEMPT_FAILED" in event_types, (
        f"expected a RELEASE_ATTEMPT_FAILED event to be logged, got {event_types}"
    )
    for event in trail:
        assert real_otp not in str(event["payload"]), (
            "the real OTP leaked into the audit log via a failed-attempt event"
        )
        assert wrong_otp not in str(event["payload"]), (
            "the attempted (wrong) OTP leaked into the audit log"
        )
