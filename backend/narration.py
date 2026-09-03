"""
narration.py — Turns top-SHAP features into a plain-language explanation.

WHY THIS IS A SEPARATE MODULE, AND WHY IT IS NEVER ON THE CRITICAL PATH:
The score and the decision (cooling-off / advisory / none) are computed and
finalized BEFORE this module is ever called. Narration only makes the
existing decision easier for a human to read — it never influences the
probability, the threshold comparison, or the action. If both free LLM
tiers are down or rate-limited during a live demo, the app still works:
it just falls back to a deterministic, template-based sentence built
directly from the SHAP features. That fallback is not a degraded mode to
apologize for — it's the same explanation logic in different clothing, and
it's actually MORE auditable than an LLM sentence because it's fully
reproducible from the inputs.

Providers (both free, no card required):
  1. Google Gemini (AI Studio) — gemini-1.5-flash / gemini-2.0-flash, free
     tier, primary because it's fast and generous for a solo/no-budget build.
  2. Groq — free tier, Llama/OSS models, used only if Gemini errors or 429s.
  3. Deterministic template — always available, zero network calls.

Each network attempt is wrapped in its own short timeout so a hung request
can't stall the demo; total narration budget is capped (NARRATION_TIMEOUT_S).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeoutError
from typing import Any, Optional

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

NARRATION_TIMEOUT_S = float(os.environ.get("NARRATION_TIMEOUT_S", "4"))

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="narration")


# --------------------------------------------------------------------------- #
# Deterministic fallback — always correct, always available
# --------------------------------------------------------------------------- #

def _humanize_feature(feature_name: str, value: Any, contribution: float) -> str:
    """Best-effort plain-English phrasing for one SHAP feature."""
    lookup = {
        "is_new_payee": "new payee (first transaction to this VPA)",
        "call_overlap": "a phone call with an unknown number overlapping the payment",
        "amount_ratio_to_avg": "an amount far above your typical spend",
        "device_change": "a payment made from an unrecognized device",
        "txn_hour_unusual": "a transaction time outside your normal pattern",
        "payee_reported_count": "this payee has prior fraud reports",
        "sim_swap_recent": "a recent SIM swap on the account",
        "log_amount": "a payment amount well above what this person normally sends",
        "amount": "a payment amount well above what this person normally sends",
        "call_overlap_flag": "an active phone call was happening during the payment",
        "call_minutes": "an unusually long phone call during the payment",
        "payee_novelty_days": "this is a brand-new or rarely-used recipient",
        "payee_account_age_days": "the recipient's account is very new",
        "payee_name_match_score": "the recipient's name does not closely match their registered UPI name",
        "hour_of_day": "the payment happened at an unusual time of day",
        "amount_p95_ratio": "the amount is far above this person's typical high-end spending",
        "roundness_score": "the amount is a suspiciously round number",
        "fresh_device_flag": "a recently reinstalled or new device",
        "screen_share_flag": "a screen-sharing session was active",
        "otp_share_flag": "a one-time password may have been shared",
        "session_language_mismatch_flag": "the session language did not match the user's usual language",
        "user_txn_count_last_1h": "several payments happened in a short window",
        "user_txn_count_last_24h": "several payments happened in a short window",
        "time_since_last_txn_hours": "this followed very soon after another payment",
        "day_of_week": "minor contextual factors about when the payment happened",
        "shopping_category_code": "minor contextual factors about the payment category",
        "instrument_type_code": "minor contextual factors about the payment method",
    }
    if feature_name in lookup:
        return lookup[feature_name]
    return "a combination of smaller contextual signals"


def deterministic_template(
    top_shap_features: list[dict], probability: float, action_type: str
) -> str:
    """
    Build a template sentence directly from the top-3 SHAP contributions.
    No network call, fully deterministic, always reproducible from the
    same inputs — useful for the audit trail as well as the UI.
    """
    phrases = [
        _humanize_feature(f["feature"], f.get("value"), f["contribution"])
        for f in top_shap_features[:3]
    ]
    joined = "; ".join(phrases) if phrases else "no single dominant factor"

    if action_type == "cooling_off":
        lead = f"Flagged (risk score {probability:.0%}) primarily due to: {joined}."
    elif action_type == "advisory_only":
        lead = (
            f"Risk score {probability:.0%} crossed the threshold, but you've "
            f"reached today's cooling-off limit, so this is a heads-up only: {joined}."
        )
    else:
        lead = f"Risk score {probability:.0%} — below the action threshold. Key factors: {joined}."
    return lead


# --------------------------------------------------------------------------- #
# Gemini (primary)
# --------------------------------------------------------------------------- #

def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    import google.generativeai as genai  # local import: optional dependency

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 120, "temperature": 0.2},
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty text")
    return text


# --------------------------------------------------------------------------- #
# Groq (fallback)
# --------------------------------------------------------------------------- #

def _call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")
    from groq import Groq  # local import: optional dependency

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned empty text")
    return text


def _build_prompt(top_shap_features: list[dict], probability: float, action_type: str) -> str:
    feat_lines = "\n".join(
        f"- {_humanize_feature(f['feature'], f.get('value'), f['contribution'])}"
        for f in top_shap_features[:3]
    )
    return (
        "You are writing a one-sentence, plain-language explanation for a UPI "
        "payer of why their payment was flagged by a fraud-risk model. Do not "
        "recommend blocking, reversing, or contacting anyone. Only explain the "
        "risk factors in friendly, non-alarming language, under 40 words.\n\n"
        f"Model risk score: {probability:.2%}\n"
        f"System action taken: {action_type}\n"
        f"Top contributing factors:\n{feat_lines}\n"
    )


# --------------------------------------------------------------------------- #
# Orchestrator — the only function serve.py should call
# --------------------------------------------------------------------------- #

def get_narration(
    top_shap_features: list[dict],
    probability: float,
    action_type: str,
) -> dict:
    """
    Returns {"text": str, "source": "gemini" | "groq" | "template_fallback",
             "latency_ms": int}

    Guarantees:
      - Never raises.
      - Never blocks longer than ~2 * NARRATION_TIMEOUT_S.
      - Always returns a usable sentence (worst case: the template).
    Call this AFTER /score and /decision have already produced their final,
    returnable values — narration failure must never change or delay them.
    """
    start = time.monotonic()
    prompt = _build_prompt(top_shap_features, probability, action_type)

    for provider_name, fn in (("gemini", _call_gemini), ("groq", _call_groq)):
        try:
            future = _executor.submit(fn, prompt)
            text = future.result(timeout=NARRATION_TIMEOUT_S)
            return {
                "text": text,
                "source": provider_name,
                "latency_ms": int((time.monotonic() - start) * 1000),
            }
        except FutTimeoutError:
            continue  # provider too slow — try next, then template
        except Exception:
            continue  # any provider error (missing key, 429, network) — try next

    text = deterministic_template(top_shap_features, probability, action_type)
    return {
        "text": text,
        "source": "template_fallback",
        "latency_ms": int((time.monotonic() - start) * 1000),
    }
