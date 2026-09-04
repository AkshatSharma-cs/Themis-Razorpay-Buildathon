"""
test_frontend_integration.py — Integration tests verifying that frontend endpoints and preset payloads work with backend FastAPI service.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.serve as serve

HEADERS = {"X-API-Key": "themis-demo-key"}


@pytest.fixture
def client():
    with TestClient(serve.app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "threshold" in data


def test_preset_call_pressure_decision(client):
    payload = {
        "payer_vpa": "sunita.patel@okhdfcbank",
        "payee_vpa": "cyber_sec_clearance@paytm",
        "amount": 48000.0,
        "user_tenure_days": 400.0,
        "user_prior_txn_count_90d": 15.0,
        "user_mean_amt_90d": 850.0,
        "user_p95_amt_90d": 1500.0,
        "user_max_amt_90d": 5000.0,
        "fresh_device_flag": True,
        "days_since_reinstall": 0.5,
        "payee_novelty_days": 0.0,
        "user_payee_txn_count_90d": 0.0,
        "payee_velocity_24h": 18.0,
        "payee_account_age_days": 3.0,
        "payee_name_match_score": 0.1,
        "call_overlap_flag": True,
        "call_minutes": 28.5,
        "screen_share_flag": False,
        "otp_share_flag": True,
        "session_language_mismatch_flag": True,
        "hour_of_day": 21,
        "day_of_week": 4,
        "time_since_last_txn_hours": 0.2,
        "user_txn_count_last_1h": 3.0,
        "user_txn_count_last_24h": 4.0,
        "shopping_category": "other",
        "instrument_type": "upi_p2p"
    }

    res = client.post("/v1/decision", json=payload, headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "tx_id" in data
    assert data["action_type"] in ("cooling_off", "advisory_only", "none")
    assert "probability" in data
    assert "narration" in data

    tx_id = data["tx_id"]

    # Test Audit endpoint for this tx_id
    audit_res = client.get(f"/v1/audit/{tx_id}")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["tx_id"] == tx_id
    assert len(audit_data["events"]) > 0


def test_escape_path_otp_release(client):
    payload = {
        "payer_vpa": "payer.otp@okaxis",
        "payee_vpa": "payee.otp@upi",
        "amount": 50000.0,
        "user_mean_amt_90d": 500.0,
        "user_p95_amt_90d": 1500.0,
        "call_overlap_flag": True,
        "screen_share_flag": True
    }
    dec_res = client.post("/v1/decision", json=payload, headers=HEADERS)
    assert dec_res.status_code == 200
    tx_id = dec_res.json()["tx_id"]

    # Request OTP
    otp_res = client.post("/v1/release/request-otp", json={"tx_id": tx_id, "payer_vpa": "payer.otp@okaxis"}, headers=HEADERS)
    assert otp_res.status_code == 200
    otp_code = otp_res.json()["otp_demo_only"]

    # Confirm Release
    confirm_res = client.post("/v1/release/confirm", json={
        "tx_id": tx_id,
        "payer_vpa": "payer.otp@okaxis",
        "otp_code": otp_code,
        "confirm_intent": True
    }, headers=HEADERS)
    assert confirm_res.status_code == 200
    assert confirm_res.json()["released"] is True


def test_verify_audit_chain_endpoint(client):
    res = client.get("/audit/verify/chain")
    assert res.status_code == 200
    assert res.json()["ok"] is True
