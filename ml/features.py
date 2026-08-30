"""
features.py
------------
Builds the model-facing feature matrix from the OBSERVED dataset only
(data/<regime>_observed.csv). This module never imports latent_process
and never opens a `*_full_latent.csv` file -- that is the physical
enforcement of "the model can't see its own generator's ground truth".

Feature groups below are each tagged with why they matter for THIS fraud
class specifically ("authorized but deceived" UPI scams), not generic
anomaly detection. A generic AML/anomaly model would lean hard on device/
IP/velocity signals because it's built for account-takeover fraud, where
the attacker is a stranger acting through the victim's channel. Here the
victim is authenticating and paying themselves, in real time, under active
social engineering -- so the useful tells are mostly about the SESSION
(is someone actively talking them through it?) and the PAYEE (is this a
brand-new, high-throughput recipient?), not the device.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that would act as label caches if left in -- a tree model would
# happily memorize "this exact user_id/payee_id was fraud before" instead
# of learning generalizable risk signal, and it would fall apart on any
# user/payee it hasn't seen (which, by construction of our entity-disjoint
# split in train.py, is every user/payee in the test set).
ID_COLUMNS_TO_DROP = ["txn_id", "user_id", "payee_id", "payee_vpa"]

ROUND_SUM_ANCHORS = np.array([500, 1000, 2000, 5000, 10000, 25000, 50000, 100000])

CATEGORICAL_COLUMNS = ["shopping_category", "instrument_type"]


def _roundness_score(amount: np.ndarray) -> np.ndarray:
    """Continuous 'how suspiciously round is this number' signal: closeness
    (in log space) to the nearest anchor in a fixed set of round sums.
    Scammers push victims toward round figures (\u20b950,000 / \u20b91,00,000); organic
    retail UPI spend rarely lands exactly on one. 0 = far from any round
    anchor, 1 = sits exactly on one."""
    log_amt = np.log10(np.maximum(amount, 1))
    log_anchors = np.log10(ROUND_SUM_ANCHORS)
    dist = np.min(np.abs(log_amt[:, None] - log_anchors[None, :]), axis=1)
    return np.exp(-8.0 * dist)  # decays fast away from an anchor


def build_features(observed: pd.DataFrame, is_train: bool = False,
                    category_levels: dict | None = None):
    """Turn an OBSERVED dataframe into (X, y, category_levels).

    `category_levels` should be None on the first (train) call -- it will
    be learned and returned. Pass the returned dict back in on subsequent
    (dev/test/cross-process) calls so category codes stay consistent
    across splits (a category unseen in train becomes an explicit
    'unknown' level rather than silently shifting every other code).
    """
    df = observed.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    y = df["label"].astype(int).values if "label" in df.columns else None

    feats = pd.DataFrame(index=df.index)

    # ---- User identity/context -------------------------------------
    # WHY: an account with long tenure and a stable spend history has a
    # meaningful "normal" to deviate from; a brand-new / rarely-active
    # account doesn't, and separately, recent reinstall/device-rebind
    # activity is a (weak, see note below) takeover-adjacent proxy.
    feats["user_tenure_days"] = df["user_tenure_days"]
    feats["user_prior_txn_count_90d"] = df["user_prior_txn_count_90d"]
    feats["user_mean_amt_90d"] = df["user_mean_amt_90d"]
    feats["user_p95_amt_90d"] = df["user_p95_amt_90d"]
    feats["user_max_amt_90d"] = df["user_max_amt_90d"]
    # NOTE (per spec): device-change is a WEAK signal for THIS fraud class
    # specifically, because in an "authorized but deceived" scam the
    # victim uses their OWN device throughout -- there is no takeover.
    # We keep it only as a fresh-install/rebind proxy (occasionally
    # correlated with a victim setting up UPI for the first time right
    # before being coerced), not as a primary flag, and we do not engineer
    # any interaction terms that would let the model lean on it harder
    # than the raw signal supports.
    feats["fresh_device_flag"] = df["fresh_device_flag"].astype(int)
    feats["days_since_reinstall"] = df["days_since_reinstall"].fillna(9999.0)

    # ---- Payee side --------------------------------------------------
    # WHY: the single most decisive structural fact about this fraud class
    # is WHO the money is going to -- a brand-new payee, rarely used by
    # this user before, but suddenly being paid by many different users in
    # a short window, is the classic mule-account signature. Account age
    # and name-match catch spoofed/rented collection accounts.
    feats["payee_novelty_days"] = df["payee_novelty_days"]
    feats["user_payee_txn_count_90d"] = df["user_payee_txn_count_90d"]
    feats["payee_velocity_24h"] = df["payee_velocity_24h"]
    feats["payee_account_age_days"] = df["payee_account_age_days"]
    feats["payee_name_match_score"] = df["payee_name_match_score"]

    # ---- Session/context: the merchant-side differentiator -----------
    # WHY: this is what actually distinguishes "authorized but deceived"
    # from every other UPI fraud pattern -- an active call, a screen-share
    # / remote-support session, or an OTP/PIN disclosure overlapping with
    # the payment is the direct behavioral fingerprint of live social
    # engineering, something account-takeover or stolen-card fraud doesn't
    # produce (there, the victim isn't present and coached in real time).
    feats["call_overlap_flag"] = df["call_overlap_flag"].astype(int)
    feats["call_minutes"] = df["call_minutes"]
    feats["screen_share_flag"] = df["screen_share_flag"].astype(int)
    feats["otp_share_flag"] = df["otp_share_flag"].astype(int)
    feats["session_language_mismatch_flag"] = df["session_language_mismatch_flag"].astype(int)

    # ---- Transaction shape --------------------------------------------
    # WHY: scam payments tend to be large relative to the victim's own
    # history and suspiciously round (a fraudster asks for "50,000", a
    # grocery run does not land on a round number by chance). Velocity
    # features catch a user being walked through several payments in a
    # short window, a common multi-step coercion pattern ("send this
    # first to verify your account, now send the rest").
    amount = df["amount"].values.astype(float)
    p95 = df["user_p95_amt_90d"].values.astype(float)
    feats["amount"] = amount
    feats["log_amount"] = np.log1p(amount)
    feats["amount_p95_ratio"] = amount / np.maximum(p95, 100.0)  # floor avoids /~0 blowups for brand-new users
    feats["roundness_score"] = _roundness_score(amount)
    feats["hour_of_day"] = df["hour_of_day"]
    feats["day_of_week"] = df["day_of_week"]  # correlated-but-non-causal: kept, but see train.py SHAP check
    feats["time_since_last_txn_hours"] = df["time_since_last_txn_hours"]
    feats["user_txn_count_last_1h"] = df["user_txn_count_last_1h"]
    feats["user_txn_count_last_24h"] = df["user_txn_count_last_24h"]

    # ---- Categorical columns (label-encoded, consistent across splits) --
    # shopping_category is the OTHER correlated-but-non-causal feature
    # named in the spec: it co-varies with coercion context somewhat (scam
    # payees get mislabeled as "person_to_person"/"other" more often) but
    # is not, on its own, decisive -- a model that leans heavily on this
    # column alone should show up as a red flag in the SHAP leakage check
    # in train.py.
    if category_levels is None:
        category_levels = {}
    for col in CATEGORICAL_COLUMNS:
        if is_train or col not in category_levels:
            levels = sorted(df[col].astype(str).unique().tolist())
            category_levels[col] = {lvl: i for i, lvl in enumerate(levels)}
            category_levels[col]["__unknown__"] = len(levels)
        mapping = category_levels[col]
        unk = mapping["__unknown__"]
        feats[f"{col}_code"] = df[col].astype(str).map(mapping).fillna(unk).astype(int)

    # ---- ID columns: explicitly dropped, never engineered from --------
    # (they were never added to `feats` in the first place -- listed here
    # only so a reviewer can see the drop is deliberate and documented)
    assert all(c not in feats.columns for c in ID_COLUMNS_TO_DROP), (
        "ID columns leaked into the feature matrix -- this would let the "
        "model memorize identities instead of learning risk signal."
    )

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(feats.median(numeric_only=True))

    meta = df[["txn_id", "user_id", "payee_id", "timestamp", "amount"]].copy()
    return feats, y, category_levels, meta


FEATURE_GROUPS = {
    "user_identity_context": ["user_tenure_days", "user_prior_txn_count_90d",
                               "user_mean_amt_90d", "user_p95_amt_90d", "user_max_amt_90d",
                               "fresh_device_flag", "days_since_reinstall"],
    "payee_side": ["payee_novelty_days", "user_payee_txn_count_90d", "payee_velocity_24h",
                    "payee_account_age_days", "payee_name_match_score"],
    "session_context": ["call_overlap_flag", "call_minutes", "screen_share_flag",
                         "otp_share_flag", "session_language_mismatch_flag"],
    "transaction_shape": ["amount", "log_amount", "amount_p95_ratio", "roundness_score",
                           "hour_of_day", "day_of_week", "time_since_last_txn_hours",
                           "user_txn_count_last_1h", "user_txn_count_last_24h",
                           "shopping_category_code", "instrument_type_code"],
}


if __name__ == "__main__":
    df = pd.read_csv("data/regime_a_observed.csv")
    X, y, levels, meta = build_features(df, is_train=True)
    print("Feature matrix shape:", X.shape)
    print("Columns:", X.columns.tolist())
    print("Positive rate:", y.mean())
    assert not any(c in X.columns for c in ID_COLUMNS_TO_DROP)
    print("ID-leakage assertion passed: no id columns in feature matrix.")
