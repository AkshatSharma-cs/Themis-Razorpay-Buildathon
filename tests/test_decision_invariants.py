"""
tests/test_decision_invariants.py

Covers the single most important invariant in the codebase (per
ARCHITECTURE.md section 2): Sentinel can never emit an action that blocks,
reverses, freezes, or otherwise acts against a payee -- only
cooling_off / advisory_only / none, and never with forbidden language in
the human-readable explanation.
"""
from __future__ import annotations

import pytest

import backend.serve as serve
from backend.serve import DefenseAction, TransactionPayload

ALLOWED_ACTION_TYPES = {"cooling_off", "advisory_only", "none"}
FORBIDDEN_TERMS = ["block", "reverse", "chargeback", "freeze", "payee_account"]


# --------------------------------------------------------------------------- #
# 1. Only the three allowed action_types can ever be constructed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad_type", [
    "block", "reverse_payment", "chargeback", "freeze_account",
    "payee_account_debit", "escalate_to_bank", "", "COOLING_OFF",  # case-sensitive
])
def test_illegal_action_type_is_rejected(bad_type):
    with pytest.raises(ValueError, match="SENTINEL INVARIANT VIOLATION"):
        DefenseAction(action_type=bad_type, reason="any reason", duration_hours=1.0)


@pytest.mark.parametrize("action_type,ctor_kwargs", [
    ("cooling_off", {"reason": "high risk", "duration_hours": 2.0}),
    ("advisory_only", {"reason": "daily cap reached"}),
    ("none", {}),
])
def test_allowed_action_types_construct_cleanly(action_type, ctor_kwargs):
    ctor = getattr(DefenseAction, action_type)
    action = ctor(**ctor_kwargs)
    assert action.action_type == action_type, (
        f"constructor DefenseAction.{action_type}(...) produced an action "
        f"with action_type={action.action_type!r} instead of {action_type!r}"
    )
    assert action.action_type in ALLOWED_ACTION_TYPES


def test_cooling_off_requires_positive_duration():
    with pytest.raises(ValueError, match="cooling_off requires a positive duration_hours"):
        DefenseAction(action_type="cooling_off", reason="high risk", duration_hours=0.0)
    with pytest.raises(ValueError, match="cooling_off requires a positive duration_hours"):
        DefenseAction(action_type="cooling_off", reason="high risk", duration_hours=-1.0)


# --------------------------------------------------------------------------- #
# 2. Forbidden-term scan on reason / verification text
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("term", FORBIDDEN_TERMS)
def test_forbidden_term_in_reason_is_rejected(term):
    reason = f"we decided to {term} this transaction to protect the user"
    with pytest.raises(ValueError, match="forbidden term"):
        DefenseAction(action_type="advisory_only", reason=reason)


@pytest.mark.parametrize("term", FORBIDDEN_TERMS)
def test_forbidden_term_in_verification_is_rejected(term):
    with pytest.raises(ValueError, match="forbidden term"):
        DefenseAction(
            action_type="cooling_off",
            reason="high risk score",
            duration_hours=1.0,
            verification=f"contact support to {term} the payment",
        )


@pytest.mark.parametrize("term", FORBIDDEN_TERMS)
def test_forbidden_term_is_case_insensitive(term):
    """The scan lowercases the combined text before matching -- confirm
    it can't be bypassed by capitalizing the forbidden word."""
    reason = f"We will {term.upper()} this transaction."
    with pytest.raises(ValueError, match="forbidden term"):
        DefenseAction(action_type="advisory_only", reason=reason)


def test_clean_reason_with_no_forbidden_terms_is_accepted():
    action = DefenseAction.advisory_only(
        reason="Risk score exceeded the threshold; showing a heads-up notice."
    )
    assert action.action_type == "advisory_only"


# --------------------------------------------------------------------------- #
# 3. End-to-end: serve.py's score -> decision flow can NEVER emit an
#    illegal action, across a spread of probability inputs straddling the
#    threshold.
# --------------------------------------------------------------------------- #

@pytest.fixture
def isolated_bundle(monkeypatch, tmp_path):
    """
    Point serve.bundle at a fixed, known threshold and route every
    audit_db call made from inside serve.py at a private temp SQLite
    file, so this test doesn't depend on (or pollute) any real model
    artifact or the default audit DB location.

    append_event()/count_events_today()'s `db_path` default is bound at
    function-definition time (module import time), so monkeypatching
    `audit_db.DB_PATH` afterwards would NOT change what serve.py's calls
    actually write to -- we replace the functions themselves instead.
    """
    import backend.audit_db as audit_db

    temp_db = str(tmp_path / "audit_test.sqlite3")
    audit_db.init_db(db_path=temp_db)

    # IMPORTANT: capture the ORIGINAL functions before patching. serve.audit_db
    # and this test's `audit_db` are the same module object, so a lambda that
    # closes over the name `audit_db.append_event` (looked up at *call* time,
    # after the patch is applied) would call itself forever.
    original_append_event = audit_db.append_event
    original_count_events_today = audit_db.count_events_today

    monkeypatch.setattr(
        serve.audit_db, "append_event",
        lambda *a, **kw: original_append_event(*a, **{**kw, "db_path": temp_db}),
    )
    monkeypatch.setattr(
        serve.audit_db, "count_events_today",
        lambda *a, **kw: original_count_events_today(*a, **{**kw, "db_path": temp_db}),
    )

    serve.bundle.metrics = {"threshold": 0.5}
    return temp_db


def _fixed_score(probability):
    """Monkeypatch target: makes bundle.score() deterministic regardless
    of whatever model (real or dummy) happens to be loaded."""
    def _score(payload):
        return probability, [
            {"feature": "amount_p95_ratio", "value": 4.2, "contribution": 0.31},
            {"feature": "payee_novelty_days", "value": 0.0, "contribution": 0.22},
            {"feature": "call_overlap_flag", "value": 1.0, "contribution": 0.18},
        ]
    return _score


@pytest.mark.parametrize("probability", [0.0, 0.49, 0.5, 0.51, 1.0])
def test_decision_action_type_always_legal(isolated_bundle, monkeypatch, probability):
    monkeypatch.setattr(serve.bundle, "score", _fixed_score(probability))

    payload = TransactionPayload(
        payer_vpa="alice@upi",
        payee_vpa="mule123@upi",
        amount=50000.0,
    )
    result = serve.decide(payload)

    assert result.action_type in ALLOWED_ACTION_TYPES, (
        f"decide() returned illegal action_type {result.action_type!r} for "
        f"probability={probability}"
    )
    # decide() builds `reason` itself -- confirm that text is also clean,
    # catching a regression where a future edit starts formatting reason
    # strings with forbidden vocabulary.
    reason_lower = result.reason.lower()
    for term in FORBIDDEN_TERMS:
        assert term not in reason_lower, (
            f"decide() produced a reason string containing forbidden term "
            f"{term!r}: {result.reason!r}"
        )


def test_decision_below_threshold_is_none(isolated_bundle, monkeypatch):
    monkeypatch.setattr(serve.bundle, "score", _fixed_score(0.1))
    payload = TransactionPayload(payer_vpa="bob@upi", payee_vpa="shop@upi", amount=200.0)
    result = serve.decide(payload)
    assert result.action_type == "none", (
        f"probability 0.1 is below the 0.5 threshold, expected action_type='none', "
        f"got {result.action_type!r}"
    )


def test_decision_above_threshold_is_cooling_off_until_daily_cap(isolated_bundle, monkeypatch):
    """Mirrors the daily-cap logic in serve.py: the first
    SENTINEL_DAILY_CAP (default 3) high-risk decisions for a payer in a
    day get cooling_off; after that, advisory_only -- never a fourth
    cooling-off, and never anything outside those two."""
    monkeypatch.setattr(serve.bundle, "score", _fixed_score(0.9))
    payer = "carol@upi"

    seen_types = []
    for i in range(5):
        payload = TransactionPayload(payer_vpa=payer, payee_vpa=f"payee{i}@upi", amount=99999.0)
        result = serve.decide(payload)
        seen_types.append(result.action_type)

    assert seen_types[:3] == ["cooling_off"] * 3, (
        f"expected the first 3 high-risk decisions to be cooling_off, got {seen_types[:3]}"
    )
    assert all(t == "advisory_only" for t in seen_types[3:]), (
        f"expected decisions past the daily cap to be advisory_only, got {seen_types[3:]}"
    )
    assert set(seen_types) == {"cooling_off", "advisory_only"}, (
        f"daily-cap sequence produced an action_type outside the allowed set: {set(seen_types)}"
    )
