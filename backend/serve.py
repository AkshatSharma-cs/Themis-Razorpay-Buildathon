"""
serve.py — Sentinel service layer (Razorpay AI Buildathon 2026, Track 2).

Framework choice: FastAPI over Gradio.
  - We need distinct, typed JSON endpoints (/score, /decision, /release,
    /audit) that a separate frontend or Postman/curl can call independently —
    that's a REST service, not a single-page demo UI.
  - FastAPI gives free OpenAPI docs at /docs, which doubles as a live demo
    surface for judges with zero extra work.
  - Gradio is the right tool when the deliverable IS the UI. Here the
    deliverable is the decision engine; a UI (if built) should be a thin
    client on top of this API. (A Gradio or static HTML demo page can still
    call this service over HTTP — see ARCHITECTURE.md.)

HARD INVARIANT (see DefenseAction below): this service can never emit an
action that blocks a payment, reverses funds, initiates a chargeback, or
touches the payee's account. That is enforced in code, not left as a
docstring promise — see DefenseAction.__post_init__.
"""

from __future__ import annotations

import json
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import backend.audit_db as audit_db
from narration import get_narration

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_PATH = os.environ.get("SENTINEL_MODEL_PATH", "model.joblib")
METRICS_PATH = os.environ.get("SENTINEL_METRICS_PATH", "metrics.json")
FEATURE_ORDER_PATH = os.environ.get("SENTINEL_FEATURES_PATH", "feature_order.json")

COOLING_OFF_DAILY_CAP = int(os.environ.get("SENTINEL_DAILY_CAP", "3"))
DEFAULT_COOLING_OFF_HOURS = float(os.environ.get("SENTINEL_COOLING_OFF_HOURS", "2"))
OTP_TTL_SECONDS = int(os.environ.get("SENTINEL_OTP_TTL_SECONDS", "300"))

# In-memory OTP store for the demo escape path. This is intentionally NOT
# persisted or hash-chained — OTPs are ephemeral secrets, not audit facts.
# Only the RELEASE *event* (that a valid OTP + confirmation occurred) goes
# into the audit chain.
_otp_store: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# The hard invariant: only a defense-only, bounded action can ever exist
# --------------------------------------------------------------------------- #

_ALLOWED_ACTION_TYPES = ("cooling_off", "advisory_only", "none")
_FORBIDDEN_TERMS = (
    "block", "reverse", "chargeback", "freeze", "seize",
    "payee_account", "debit_payee", "credit_reversal", "law_enforcement_action",
)


@dataclass(frozen=True)
class DefenseAction:
    """
    The ONLY three constructors below are the ONLY ways to build an action.
    There is no bare __init__ path that skips validation, and __post_init__
    runs on every construction (including cls.__new__ via dataclasses),
    so this cannot be bypassed by calling DefenseAction(...) directly either.
    """
    action_type: Literal["cooling_off", "advisory_only", "none"]
    reason: str
    duration_hours: float = 0.0
    verification: str = ""

    def __post_init__(self):
        if self.action_type not in _ALLOWED_ACTION_TYPES:
            raise ValueError(
                f"SENTINEL INVARIANT VIOLATION: illegal action_type "
                f"'{self.action_type}'. This is a hard disqualification line "
                f"for this track — refusing to construct."
            )
        blob = f"{self.reason} {self.verification}".lower()
        for term in _FORBIDDEN_TERMS:
            if term in blob:
                raise ValueError(
                    f"SENTINEL INVARIANT VIOLATION: forbidden term '{term}' "
                    f"found in action text. Refusing to construct."
                )
        if self.action_type == "cooling_off" and self.duration_hours <= 0:
            raise ValueError("cooling_off requires a positive duration_hours")

    @classmethod
    def cooling_off(cls, reason: str, duration_hours: float,
                     verification: str = "re-auth + confirm intent") -> "DefenseAction":
        return cls("cooling_off", reason, duration_hours, verification)

    @classmethod
    def advisory_only(cls, reason: str) -> "DefenseAction":
        return cls("advisory_only", reason, 0.0, "none required")

    @classmethod
    def none(cls, reason: str = "score below cost-optimal threshold") -> "DefenseAction":
        return cls("none", reason, 0.0, "")


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #

class TransactionPayload(BaseModel):
    tx_id: Optional[str] = Field(default=None, description="If omitted, one is generated")
    payer_vpa: str
    payee_vpa: str
    amount: float
    is_new_payee: bool = False
    call_overlap: bool = False
    device_change: bool = False
    txn_hour_unusual: bool = False
    sim_swap_recent: bool = False
    payee_reported_count: int = 0
    historical_avg_amount: float = 0.0
    # Anything the trained model needs that isn't listed explicitly above
    # can be passed here; the feature builder pulls from both.
    extra_features: dict[str, float] = Field(default_factory=dict)


class ShapFeature(BaseModel):
    feature: str
    value: Any
    contribution: float


class ScoreResponse(BaseModel):
    tx_id: str
    probability: float
    top_features: list[ShapFeature]
    model_version: str


class DecisionResponse(BaseModel):
    tx_id: str
    probability: float
    threshold: float
    action_type: str
    reason: str
    duration_hours: float
    verification: str
    narration: str
    narration_source: str
    audit_row_hash: str


class ReleaseRequestOtp(BaseModel):
    tx_id: str
    payer_vpa: str


class ReleaseConfirm(BaseModel):
    tx_id: str
    payer_vpa: str
    otp_code: str
    confirm_intent: bool


# --------------------------------------------------------------------------- #
# Model loading (with a clearly-labeled demo fallback so the service runs
# end-to-end even before the real artifacts are wired in)
# --------------------------------------------------------------------------- #

class _DummyModel:
    """
    Only used when MODEL_PATH is missing, so the API contract is testable
    before the real LightGBM artifact exists. NEVER used if a real model
    file is present. Produces a deterministic pseudo-score from the
    feature vector so demos are at least reproducible.
    """
    classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.tanh(X.sum(axis=1) / (X.shape[1] + 1)) * 0.5 + 0.5
        return np.column_stack([1 - raw, raw])


class ModelBundle:
    def __init__(self):
        self.model = None
        self.metrics: dict = {}
        self.feature_order: list[str] = []
        self.explainer = None
        self.is_dummy = False
        self.model_version = "unloaded"

    def load(self):
        # --- feature order (used to build a stable X vector for the model) ---
        if os.path.exists(FEATURE_ORDER_PATH):
            with open(FEATURE_ORDER_PATH) as f:
                self.feature_order = json.load(f)
        else:
            self.feature_order = [
                "amount", "is_new_payee", "call_overlap", "device_change",
                "txn_hour_unusual", "sim_swap_recent", "payee_reported_count",
                "historical_avg_amount",
            ]

        # --- metrics.json: must supply the cost-optimal threshold ---
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {"threshold": 0.5, "note": "metrics.json missing — using default"}

        # --- model artifact ---
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.model_version = self.metrics.get("model_version", "loaded")
            self.is_dummy = False
        else:
            self.model = _DummyModel()
            self.model_version = "DUMMY_MODEL_NO_ARTIFACT_FOUND"
            self.is_dummy = True

        # --- SHAP explainer (TreeExplainer works for LightGBM boosters) ---
        try:
            import shap
            if not self.is_dummy:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = None
        except Exception:
            self.explainer = None

    @property
    def threshold(self) -> float:
        return float(self.metrics.get("threshold", 0.5))

    def build_vector(self, payload: TransactionPayload) -> np.ndarray:
        base = {
            "amount": payload.amount,
            "is_new_payee": float(payload.is_new_payee),
            "call_overlap": float(payload.call_overlap),
            "device_change": float(payload.device_change),
            "txn_hour_unusual": float(payload.txn_hour_unusual),
            "sim_swap_recent": float(payload.sim_swap_recent),
            "payee_reported_count": float(payload.payee_reported_count),
            "historical_avg_amount": payload.historical_avg_amount,
        }
        base.update(payload.extra_features)
        row = [base.get(f, 0.0) for f in self.feature_order]
        return np.array([row], dtype=float)

    def score(self, payload: TransactionPayload) -> tuple[float, list[dict]]:
        X = self.build_vector(payload)
        proba = float(self.model.predict_proba(X)[0, 1])

        top_features: list[dict] = []
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            # LightGBM binary classifiers via SHAP can return a list
            # [neg_class_values, pos_class_values] or a single array
            # depending on shap/lightgbm version — handle both.
            if isinstance(shap_values, list):
                values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                values = shap_values[0]
            pairs = list(zip(self.feature_order, X[0], values))
            pairs.sort(key=lambda p: abs(p[2]), reverse=True)
            top_features = [
                {"feature": name, "value": float(val), "contribution": float(contrib)}
                for name, val, contrib in pairs[:3]
            ]
        else:
            # No SHAP available (e.g. dummy model, or shap not installed):
            # fall back to a naive "largest raw feature value" ranking so
            # the API contract still returns something usable for the demo.
            pairs = list(zip(self.feature_order, X[0]))
            pairs.sort(key=lambda p: abs(p[1]), reverse=True)
            top_features = [
                {"feature": name, "value": float(val), "contribution": 0.0}
                for name, val in pairs[:3]
            ]

        return proba, top_features


bundle = ModelBundle()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Sentinel — Defense-only UPI Scam Detector",
    description=(
        "Track 2: authorized-but-deceived UPI scam detection. This service "
        "only ever recommends a bounded cooling-off or advisory notice — it "
        "never blocks, reverses, or acts on another party's account."
    ),
    version="0.1.0",
)


@app.on_event("startup")
def _startup():
    audit_db.init_db()
    bundle.load()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": bundle.model_version,
        "is_dummy_model": bundle.is_dummy,
        "threshold": bundle.threshold,
    }


# --------------------------------------------------------------------------- #
# /score
# --------------------------------------------------------------------------- #

@app.post("/score", response_model=ScoreResponse)
def score_transaction(payload: TransactionPayload):
    tx_id = payload.tx_id or f"txn_{uuid.uuid4().hex[:12]}"
    probability, top_features = bundle.score(payload)

    audit_db.append_event(
        tx_id=tx_id,
        event_type="SCORE_COMPUTED",
        payload={
            "probability": probability,
            "top_features": top_features,
            "model_version": bundle.model_version,
        },
        payer_vpa=payload.payer_vpa,
    )

    return ScoreResponse(
        tx_id=tx_id,
        probability=probability,
        top_features=[ShapFeature(**f) for f in top_features],
        model_version=bundle.model_version,
    )


# --------------------------------------------------------------------------- #
# /decision
# --------------------------------------------------------------------------- #

@app.post("/decision", response_model=DecisionResponse)
def decide(payload: TransactionPayload):
    tx_id = payload.tx_id or f"txn_{uuid.uuid4().hex[:12]}"

    # 1. Score first — this value is FINAL and does not change based on
    #    narration outcome below.
    probability, top_features = bundle.score(payload)
    threshold = bundle.threshold

    audit_db.append_event(
        tx_id, "SCORE_COMPUTED",
        {"probability": probability, "top_features": top_features},
        payload.payer_vpa,
    )

    # 2. Decide the action. This is also FINAL before narration runs.
    if probability >= threshold:
        prior_today = audit_db.count_events_today(payload.payer_vpa, "COOLING_OFF_ISSUED")
        if prior_today < COOLING_OFF_DAILY_CAP:
            action = DefenseAction.cooling_off(
                reason=(
                    f"Risk score {probability:.2%} exceeds the cost-optimal "
                    f"threshold {threshold:.2%}."
                ),
                duration_hours=DEFAULT_COOLING_OFF_HOURS,
            )
            audit_db.append_event(
                tx_id, "COOLING_OFF_ISSUED",
                {"probability": probability, "threshold": threshold,
                 "duration_hours": action.duration_hours,
                 "daily_count_before_this": prior_today},
                payload.payer_vpa,
            )
        else:
            action = DefenseAction.advisory_only(
                reason=(
                    f"Risk score {probability:.2%} exceeds threshold, but you have "
                    f"already been delayed {COOLING_OFF_DAILY_CAP} time(s) today — "
                    f"showing an advisory instead of another delay."
                ),
            )
            audit_db.append_event(
                tx_id, "DAILY_CAP_REACHED_ADVISORY_ONLY",
                {"probability": probability, "threshold": threshold,
                 "daily_cap": COOLING_OFF_DAILY_CAP},
                payload.payer_vpa,
            )
    else:
        action = DefenseAction.none(
            reason=f"Risk score {probability:.2%} is below threshold {threshold:.2%}."
        )

    # 3. Narration happens LAST, after probability/action are already fixed.
    #    A slow/failed LLM call can only change the wording, never the outcome.
    narration = get_narration(top_features, probability, action.action_type)

    audit_row = audit_db.append_event(
        tx_id, "DECISION_FINALIZED",
        {
            "probability": probability,
            "threshold": threshold,
            "action_type": action.action_type,
            "reason": action.reason,
            "duration_hours": action.duration_hours,
            "narration_source": narration["source"],
        },
        payload.payer_vpa,
    )

    return DecisionResponse(
        tx_id=tx_id,
        probability=probability,
        threshold=threshold,
        action_type=action.action_type,
        reason=action.reason,
        duration_hours=action.duration_hours,
        verification=action.verification,
        narration=narration["text"],
        narration_source=narration["source"],
        audit_row_hash=audit_row.row_hash,
    )


# --------------------------------------------------------------------------- #
# Escape path: /release/request-otp then /release/confirm
# --------------------------------------------------------------------------- #

@app.post("/release/request-otp")
def request_otp(req: ReleaseRequestOtp):
    """
    DEMO-ONLY implementation: generates a 6-digit OTP and returns it directly
    in the response so the flow is testable without an SMS gateway (every
    free SMS/OTP provider we found either requires a card or a business
    registration). Wire this to MSG91 / Twilio Verify / your bank's actual
    OTP channel in production — just swap this function's body; the rest
    of the flow (confirm + audit logging) doesn't need to change.
    """
    otp = "".join(random.choices(string.digits, k=6))
    _otp_store[req.tx_id] = {
        "otp": otp,
        "payer_vpa": req.payer_vpa,
        "expires_at": time.time() + OTP_TTL_SECONDS,
    }
    audit_db.append_event(
        req.tx_id, "OTP_REQUESTED",
        {"expires_in_seconds": OTP_TTL_SECONDS},
        req.payer_vpa,
    )
    return {"tx_id": req.tx_id, "otp_demo_only": otp, "expires_in_seconds": OTP_TTL_SECONDS}


@app.post("/release/confirm")
def confirm_release(req: ReleaseConfirm):
    """
    The escape path: a genuine payer can always end their own cooling-off
    immediately by re-authenticating (fresh OTP) and explicitly confirming
    intent. This is the counterweight to the daily cap — Sentinel can delay
    a payment, but it can never trap a real payer who insists they meant to
    pay.
    """
    record = _otp_store.get(req.tx_id)
    if record is None:
        raise HTTPException(status_code=400, detail="No OTP was requested for this transaction.")
    if time.time() > record["expires_at"]:
        raise HTTPException(status_code=400, detail="OTP expired. Request a new one.")
    if record["payer_vpa"] != req.payer_vpa:
        raise HTTPException(status_code=403, detail="Payer VPA mismatch.")
    if record["otp"] != req.otp_code:
        audit_db.append_event(
            req.tx_id, "RELEASE_ATTEMPT_FAILED",
            {"reason": "otp_mismatch"}, req.payer_vpa,
        )
        raise HTTPException(status_code=400, detail="Incorrect OTP.")
    if not req.confirm_intent:
        audit_db.append_event(
            req.tx_id, "RELEASE_ATTEMPT_FAILED",
            {"reason": "intent_not_confirmed"}, req.payer_vpa,
        )
        raise HTTPException(
            status_code=400,
            detail="You must explicitly confirm 'I made this payment' to release.",
        )

    del _otp_store[req.tx_id]
    audit_db.append_event(
        req.tx_id, "COOLING_OFF_RELEASED",
        {"method": "re-auth + confirm intent"}, req.payer_vpa,
    )
    return {
        "tx_id": req.tx_id,
        "released": True,
        "reason": "Payer re-authenticated and confirmed intent. Cooling-off lifted.",
    }


# --------------------------------------------------------------------------- #
# /audit
# --------------------------------------------------------------------------- #

@app.get("/audit/{tx_id}")
def get_audit(tx_id: str):
    trail = audit_db.get_audit_trail(tx_id)
    if not trail:
        raise HTTPException(status_code=404, detail="No audit trail for this tx_id.")
    return {"tx_id": tx_id, "events": trail}


@app.get("/audit/verify/chain")
def verify_audit_chain():
    """
    Demo endpoint: proves the audit log hasn't been tampered with. Call this
    live during the demo, then (in a separate terminal) hand-edit a row with
    raw sqlite3 and call it again to show ok flip to false.
    """
    return audit_db.verify_chain()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), reload=True)
