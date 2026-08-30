"""
generate.py
-----------
Builds two synthetic UPI transaction datasets:

  1. the "in-process" dataset (regime A) -- used for train/dev/test via a
     temporal + entity-disjoint split (see train.py).
  2. the "cross-process" dataset (regime B) -- a DIFFERENT amount regime,
     seasonality and scam-mix ratio, scored exactly once at the very end
     by train.py, after everything else (model, calibration, threshold)
     is frozen. This is our stand-in for "the fraud patterns next quarter
     won't look identical to this quarter".

ARCHITECTURE / HONEST-EVAL NOTE:
This file imports `latent_process.generate_ground_truth`, which is the
ONLY function that decides `C` (latent coercion intensity) and `label`.
Everything in this file that looks like it "uses C" is producing an
OBSERVABLE SIDE EFFECT correlated with C (the way a scam call producing
screen-share and OTP-share events is a side effect of coercion, not
coercion itself) -- never C itself. Two files get written per regime:

  * data/<regime>_full_latent.csv   -- includes C, is_candidate, and the
    pre/post-noise label columns. FOR OUR OWN DIAGNOSTICS ONLY. This file
    is never opened by features.py or train.py.
  * data/<regime>_observed.csv      -- the only file downstream code may
    read. Contains `label` (the supervised target) and observable columns
    only. No `C`, no `is_candidate`.

This split is what lets us show a reviewer, mechanically, that the model
never had access to its own generator's internals -- not just "we didn't
pass that column", but "that column lives in a file the training code
cannot open."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta
from faker import Faker

from latent_process import generate_ground_truth, LatentProcessConfig, diagnostics

fake = Faker("en_IN")


@dataclass
class ObservableConfig:
    regime_name: str
    n_rows: int
    n_users: int
    n_payees: int
    mule_payee_frac: float          # share of payee pool that is mule-like
    start_date: str
    span_days: int
    legit_amount_mu: float           # log-space mean of legit lognormal amt
    legit_amount_sigma: float
    scam_multiplier_low: float       # scam amt = user_p95 * U(low, high)
    scam_multiplier_high: float
    round_sums: tuple = (50_000.0, 100_000.0)
    round_sum_prob: float = 0.45     # chance scam amount snaps to a round sum
    weekend_bias: float = 0.0        # +ve => candidates skew more weekend
    seed: int = 0


REGIME_A = ObservableConfig(
    regime_name="regime_a",
    n_rows=5000,
    n_users=850,
    n_payees=420,
    mule_payee_frac=0.03,
    start_date="2026-03-01",
    span_days=150,
    legit_amount_mu=np.log(450),     # median legit amt ~ INR 450
    legit_amount_sigma=0.9,
    scam_multiplier_low=2.5,
    scam_multiplier_high=9.0,
    weekend_bias=0.05,
    seed=20260829,
)

# Cross-process holdout: different amount regime (higher baseline spend,
# festive-season-like seasonality), different scam-mix ratio, different
# payee pool size -- deliberately NOT the same generative parameters.
REGIME_B_CROSS_PROCESS = ObservableConfig(
    regime_name="regime_b_cross_process",
    n_rows=1500,
    n_users=340,
    n_payees=170,
    mule_payee_frac=0.05,
    start_date="2026-09-15",         # a later, festive-shopping window
    span_days=60,
    legit_amount_mu=np.log(900),     # higher baseline spend (festive season)
    legit_amount_sigma=1.05,
    scam_multiplier_low=1.8,
    scam_multiplier_high=7.0,
    weekend_bias=0.15,
    seed=20261225,
)

LATENT_CFG_A = LatentProcessConfig(candidate_rate=0.042, seed=910260829)
LATENT_CFG_B = LatentProcessConfig(candidate_rate=0.027, seed=910261225)  # different scam-mix ratio
# NOTE: these seeds are DELIBERATELY different from REGIME_A.seed /
# REGIME_B_CROSS_PROCESS.seed above. Using the same seed value to build
# two independent `np.random.default_rng` streams that are then each
# used for an early threshold-style draw (is_candidate here, era there)
# makes the underlying uniform sequences align and silently correlates
# them -- we hit this directly in development: every single
# is_candidate=True row was landing in the "fit" era, because the same
# low uniform floats that cleared the candidate_rate threshold also
# cleared the "fit" bucket of the era distribution. Different seeds
# break that accidental coupling.

SHOPPING_CATEGORIES = ["groceries", "food_delivery", "utility_bill", "person_to_person",
                        "recharge", "e_commerce", "travel", "other"]
INSTRUMENTS = ["upi_p2p", "upi_p2m", "upi_autopay"]


def _zipf_popularity(n: int, rng: np.random.Generator, exponent: float = 0.35) -> np.ndarray:
    """Popularity weights so some users/payees are more active than
    others, like real transaction data -- but with a flattened exponent
    (0.35, not the "pure" 1.0 harmonic Zipf). A pure 1/rank law
    concentrates a dominant share of ALL transaction volume onto a
    handful of accounts; at the population sizes we can afford in a
    synthetic dataset this means those few "super accounts" are active
    across the entire timeline almost by definition, which silently
    destroys the entity-disjoint split. Flattening the exponent keeps a
    realistic long tail without letting a few IDs dominate every era."""
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, exponent)
    return weights / weights.sum()


def _assign_eras(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each row to fit/dev/test BEFORE any user/payee/timestamp is
    chosen. This is what makes entity-disjointness a guarantee of the
    generator, not a hope enforced by discarding rows after the fact:
    each era gets its own slice of the user pool, its own slice of the
    payee pool (including its own slice of the mule pool), and its own
    (non-overlapping, ordered) time window. A user or payee simply never
    has the opportunity to appear in two eras, because it was never
    assigned to more than one."""
    eras = np.array(["fit", "dev", "test"])
    probs = np.array([0.65, 0.15, 0.20])
    return rng.choice(eras, size=n_rows, p=probs)


def _partition_pool(n_items: int, rng: np.random.Generator, fracs=(0.65, 0.15, 0.20)):
    """Split index range [0, n_items) into three disjoint id pools sized
    (approximately) by `fracs`, in random order (not contiguous blocks --
    contiguous blocks would accidentally correlate id number with era)."""
    idx = rng.permutation(n_items)
    n_fit = max(1, int(round(n_items * fracs[0])))
    n_dev = max(1, int(round(n_items * fracs[1])))
    fit_ids = idx[:n_fit]
    dev_ids = idx[n_fit:n_fit + n_dev]
    test_ids = idx[n_fit + n_dev:]
    if len(test_ids) == 0:  # guard tiny pools
        test_ids = dev_ids[-1:]
        dev_ids = dev_ids[:-1]
    return {"fit": fit_ids, "dev": dev_ids, "test": test_ids}


def _assign_users_and_payees(n_rows: int, cfg: ObservableConfig,
                              is_candidate: np.ndarray, c: np.ndarray,
                              era: np.ndarray, rng: np.random.Generator):
    n_mule = max(1, int(round(cfg.n_payees * cfg.mule_payee_frac)))
    mule_ids_all = np.arange(cfg.n_payees - n_mule, cfg.n_payees)   # top-index slice = mule pool
    normal_ids_all = np.arange(0, cfg.n_payees - n_mule)

    user_pools = _partition_pool(cfg.n_users, rng)
    normal_payee_pools = _partition_pool(len(normal_ids_all), rng)
    mule_payee_pools = _partition_pool(len(mule_ids_all), rng) if n_mule >= 3 else \
        {"fit": np.arange(n_mule), "dev": np.arange(n_mule), "test": np.arange(n_mule)}

    user_ids = np.empty(n_rows, dtype=int)
    payee_ids = np.empty(n_rows, dtype=int)

    mule_draw_prob = np.where(is_candidate, np.clip(0.25 + 0.6 * c, 0, 0.9), 0.01)
    use_mule = rng.random(n_rows) < mule_draw_prob

    for era_name in ("fit", "dev", "test"):
        mask = era == era_name
        n_era = int(mask.sum())
        if n_era == 0:
            continue
        era_users = user_pools[era_name]
        era_pop = _zipf_popularity(len(era_users), rng, exponent=0.35)
        user_ids[mask] = era_users[rng.choice(len(era_users), size=n_era, p=era_pop)]

        era_normal = normal_ids_all[normal_payee_pools[era_name]]
        era_mule = mule_ids_all[mule_payee_pools[era_name]]
        era_normal_pop = _zipf_popularity(len(era_normal), rng, exponent=0.35)

        # WHY "favorite payees" exist: real UPI usage is dominated by
        # recurring relationships (the same grocery store, the same
        # utility biller, the same landlord) -- without modelling that,
        # almost every (user, payee) pair in a short synthetic window is
        # a first-ever pairing, which would make `payee_novelty_days`
        # (an intentionally load-bearing feature per the spec) constant
        # and useless. Each user gets 1-3 "regulars" drawn from the era's
        # normal payee pool; legit spend mostly returns to a regular,
        # scam/mule spend (by construction, drawn from the mule pool
        # instead) never does.
        era_user_local_idx = np.arange(len(era_users))
        n_favorites = rng.integers(1, 4, size=len(era_users))
        favorites_by_user = {}
        for local_i, uid in enumerate(era_users):
            favorites_by_user[uid] = rng.choice(era_normal, size=min(n_favorites[local_i], len(era_normal)),
                                                 p=era_normal_pop, replace=False)

        use_mule_era = use_mule[mask]
        era_uids = user_ids[mask]
        era_payee = np.empty(n_era, dtype=int)
        n_use_mule_era = int(use_mule_era.sum())
        if n_use_mule_era > 0:
            era_payee[use_mule_era] = rng.choice(era_mule, size=n_use_mule_era)
        legit_mask = ~use_mule_era
        n_legit = int(legit_mask.sum())
        if n_legit > 0:
            use_favorite = rng.random(n_legit) < 0.70
            legit_uids = era_uids[legit_mask]
            legit_payees = np.empty(n_legit, dtype=int)
            fresh_draw = rng.choice(era_normal, size=n_legit, p=era_normal_pop)
            for i in range(n_legit):
                if use_favorite[i]:
                    legit_payees[i] = rng.choice(favorites_by_user[legit_uids[i]])
                else:
                    legit_payees[i] = fresh_draw[i]
            era_payee[legit_mask] = legit_payees
        payee_ids[mask] = era_payee

    return user_ids, payee_ids, mule_ids_all


def _assign_timestamps(n_rows: int, cfg: ObservableConfig, is_candidate: np.ndarray,
                        c: np.ndarray, era: np.ndarray, rng: np.random.Generator) -> pd.Series:
    start = pd.Timestamp(cfg.start_date)
    t_fit_end = cfg.span_days * 0.65
    t_dev_end = cfg.span_days * 0.80
    era_range = {
        "fit": (0.0, t_fit_end),
        "dev": (t_fit_end, t_dev_end),
        "test": (t_dev_end, float(cfg.span_days)),
    }
    base_offsets = np.empty(n_rows, dtype=float)
    for era_name, (lo, hi) in era_range.items():
        mask = era == era_name
        base_offsets[mask] = rng.uniform(lo, hi, size=int(mask.sum()))

    # weak, non-decisive confound: candidates skew slightly toward weekends.
    # Clipped back into THIS ROW'S OWN era window (not the global span) so
    # the nudge can never push a row across an era time boundary -- that
    # would silently break the fit/dev/test time-ordering guarantee.
    weekend_pull = np.where(is_candidate, cfg.weekend_bias * c, 0.0)
    nudged = base_offsets + weekend_pull * rng.normal(0, 1.5, size=n_rows)
    lo_bound = np.empty(n_rows, dtype=float)
    hi_bound = np.empty(n_rows, dtype=float)
    for era_name, (lo, hi) in era_range.items():
        mask = era == era_name
        lo_bound[mask] = lo
        hi_bound[mask] = hi
    base_offsets = np.clip(nudged, lo_bound, hi_bound - 0.001)

    # hour of day: legit traffic bell-curved around daytime; candidates
    # weakly (not decisively) shifted toward odd hours as C rises
    daytime_hour = np.clip(rng.normal(14, 4, size=n_rows), 0, 23.99)
    odd_hour = rng.uniform(0, 23.99, size=n_rows)
    hour_mix = np.where(rng.random(n_rows) < np.where(is_candidate, 0.15 + 0.35 * c, 0.05),
                         odd_hour, daytime_hour)

    # combine day-offset + hour into one continuous day-unit offset BEFORE
    # clipping to the era window -- clipping days alone and adding hours
    # afterward can push a late-era row's clock time past midnight and
    # across the era boundary, which is exactly the kind of few-hour
    # overlap that would quietly break entity-disjointness.
    combined_offset = base_offsets + hour_mix / 24.0
    combined_offset = np.clip(combined_offset, lo_bound, hi_bound - 0.001)

    ts = [start + timedelta(days=float(d)) for d in combined_offset]
    return pd.to_datetime(ts)


def _sample_context_features(n_rows: int, is_candidate: np.ndarray, c: np.ndarray,
                              rng: np.random.Generator) -> pd.DataFrame:
    """The merchant-side differentiator group. All of these are OBSERVABLE
    side effects correlated with C, sampled with a non-zero baseline floor
    so that legitimate support calls / screen-shares / language differences
    also occur -- this is what keeps the classes from being perfectly
    separable on this group alone."""
    c_eff = np.where(is_candidate, c, 0.0)

    call_overlap_p = 0.03 + 0.55 * c_eff
    call_overlap_flag = rng.random(n_rows) < call_overlap_p
    call_minutes = np.where(
        call_overlap_flag,
        rng.exponential(scale=3 + 18 * c_eff, size=n_rows),
        rng.exponential(scale=1.0, size=n_rows) * (rng.random(n_rows) < 0.04),
    )

    screen_share_p = 0.015 + 0.45 * c_eff
    screen_share_flag = rng.random(n_rows) < screen_share_p

    otp_share_p = 0.01 + 0.5 * (c_eff ** 1.3)
    otp_share_flag = rng.random(n_rows) < otp_share_p

    lang_mismatch_p = 0.04 + 0.3 * c_eff
    lang_mismatch_flag = rng.random(n_rows) < lang_mismatch_p

    # fresh-install / new-device binding: intentionally WEAK signal for
    # THIS fraud class, because in "authorized but deceived" scams the
    # victim uses their OWN device -- we only include it as a takeover
    # proxy, not a primary flag. Baseline rate dominates; C contributes a
    # small bump only.
    fresh_device_p = 0.05 + 0.06 * c_eff
    fresh_device_flag = rng.random(n_rows) < fresh_device_p
    days_since_reinstall = np.where(
        fresh_device_flag, rng.exponential(scale=3.0, size=n_rows), np.nan
    )

    return pd.DataFrame({
        "call_overlap_flag": call_overlap_flag.astype(int),
        "call_minutes": np.round(call_minutes, 1),
        "screen_share_flag": screen_share_flag.astype(int),
        "otp_share_flag": otp_share_flag.astype(int),
        "session_language_mismatch_flag": lang_mismatch_flag.astype(int),
        "fresh_device_flag": fresh_device_flag.astype(int),
        "days_since_reinstall": days_since_reinstall,
    })


def _sample_payee_side_features(n_rows: int, payee_ids: np.ndarray, mule_ids: np.ndarray,
                                 is_candidate: np.ndarray, c: np.ndarray,
                                 rng: np.random.Generator) -> pd.DataFrame:
    is_mule = np.isin(payee_ids, mule_ids)
    c_eff = np.where(is_candidate, c, 0.0)

    # payee account age: mule accounts skew young
    payee_account_age = np.where(
        is_mule,
        rng.exponential(scale=12, size=n_rows) + 1,
        rng.exponential(scale=260, size=n_rows) + 20,
    )

    # name-match score (payee display name vs. UPI-registered name):
    # spoofed/mule payees skew low; legitimate payees skew high; small
    # overlap so it's not a perfect tell on its own
    name_match = np.where(
        is_mule,
        np.clip(rng.beta(2, 5, size=n_rows), 0, 1),
        np.clip(rng.beta(6, 1.5, size=n_rows), 0, 1),
    )
    # extra weak pull toward low name-match with rising C even off mule pool
    name_match = np.clip(name_match - 0.15 * c_eff, 0, 1)

    return pd.DataFrame({
        "payee_account_age_days": np.round(payee_account_age, 1),
        "payee_name_match_score": np.round(name_match, 3),
        "_is_mule_payee": is_mule,  # internal helper, dropped before saving
    })


def _sample_amounts(is_candidate: np.ndarray, c: np.ndarray, user_ids: np.ndarray,
                     timestamps: pd.Series, cfg: ObservableConfig,
                     rng: np.random.Generator) -> np.ndarray:
    """Every row first gets a 'what this user would normally spend' draw.
    Rows are then, independently of the (noisy, capped) label, given a
    probability of an amount escalation that rises with C -- NOT with the
    final label. This is a deliberate architectural choice: if we keyed
    the amount off `label` directly, the amount column would become a
    near-perfect proxy for the (noisy) label itself (a noise-flipped
    negative would still get a giveaway-huge amount, and a noise-flipped
    positive would get a boring one), which would hand the model a
    shortcut that has nothing to do with actually detecting coercion. By
    keying off C/is_candidate instead, a transaction with real coercion
    signal but a small "test" amount is possible, and a labeled scam that
    was actually a low-C/noise-flip case doesn't come with a suspicious
    amount stapled on for free -- exactly the kind of irreducible
    ambiguity the label-noise and C-ceiling design is supposed to create."""
    n = len(user_ids)
    baseline = rng.lognormal(mean=cfg.legit_amount_mu, sigma=cfg.legit_amount_sigma, size=n)

    df = pd.DataFrame({"user_id": user_ids, "timestamp": timestamps, "baseline": baseline})
    df = df.sort_values(["user_id", "timestamp"]).reset_index()
    # trailing (prior-only) rolling p95 per user over a 90-day window
    def rolling_p95(g):
        s = g.set_index("timestamp")["baseline"]
        return s.rolling("90D", closed="left").quantile(0.95)

    p95_prior = df.groupby("user_id", group_keys=False).apply(rolling_p95, include_groups=False).values
    # first-ever transaction for a user has no prior history -> fall back
    # to a generic p95 proxy so scam rows early in a user's history still
    # get a sensible multiplier
    generic_p95 = np.nanpercentile(baseline, 95)
    p95_prior = np.where(np.isnan(p95_prior), generic_p95, p95_prior)

    df["p95_prior"] = p95_prior
    df = df.sort_values("index").reset_index(drop=True)  # restore original row order

    amount = df["baseline"].values.copy()
    c_eff = np.where(is_candidate, c, 0.0)
    # escalation PROBABILITY rises with C, but even a strongly-coerced
    # session sometimes still moves a small "test" amount, and even a
    # weak/borderline session sometimes goes straight for a large one --
    # this is a probability, not a deterministic switch on the label.
    escalate_prob = np.where(is_candidate, np.clip(0.12 + 0.80 * c_eff, 0, 0.95), 0.002)
    escalate = rng.random(n) < escalate_prob

    mult = rng.uniform(cfg.scam_multiplier_low, cfg.scam_multiplier_high, size=n)
    scam_amount = df["p95_prior"].values * mult
    use_round_sum = rng.random(n) < cfg.round_sum_prob
    round_choice = rng.choice(cfg.round_sums, size=n)
    scam_amount = np.where(use_round_sum & (round_choice > scam_amount * 0.5),
                            round_choice, scam_amount)
    scam_amount = np.maximum(scam_amount, df["p95_prior"].values * 1.5)
    amount[escalate] = scam_amount[escalate]
    return np.round(amount, 2)


def _rolling_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing, prior-only per-user history features (90-day window),
    computed in strict time order per user so nothing here can see its
    own current row."""
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    def per_user(g):
        uid = g.name
        g = g.set_index("timestamp")
        amt = g["amount"]
        roll = amt.rolling("90D", closed="left")
        g["user_prior_txn_count_90d"] = roll.count()
        g["user_mean_amt_90d"] = roll.mean()
        g["user_p95_amt_90d"] = roll.quantile(0.95)
        g["user_max_amt_90d"] = roll.max()
        g["user_tenure_days"] = (g.index - g.index.min()).days.astype(float)
        g["time_since_last_txn_hours"] = g.index.to_series().diff().dt.total_seconds() / 3600.0
        roll1h = amt.rolling("1h", closed="left")
        roll24h = amt.rolling("24h", closed="left")
        g["user_txn_count_last_1h"] = roll1h.count()
        g["user_txn_count_last_24h"] = roll24h.count()
        g["user_id"] = uid
        return g.reset_index()

    out = df.groupby("user_id", group_keys=False).apply(per_user, include_groups=False)
    return out


def _rolling_payee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Payee-side velocity (across ALL users) and per-(user,payee) novelty
    / familiarity, both trailing/prior-only."""
    df = df.sort_values(["payee_id", "timestamp"]).reset_index(drop=True)

    def per_payee(g):
        pid = g.name
        g = g.set_index("timestamp")
        roll24h = g["amount"].rolling("24h", closed="left")
        g["payee_velocity_24h"] = roll24h.count()
        g["payee_id"] = pid
        return g.reset_index()

    df = df.groupby("payee_id", group_keys=False).apply(per_payee, include_groups=False)

    df = df.sort_values(["user_id", "payee_id", "timestamp"]).reset_index(drop=True)

    def per_user_payee(g):
        uid, pid = g.name
        g = g.set_index("timestamp")
        g["payee_novelty_days"] = (g.index - g.index.min()).days.astype(float)
        g["user_payee_txn_count_90d"] = g["amount"].rolling("90D", closed="left").count()
        g["user_id"] = uid
        g["payee_id"] = pid
        return g.reset_index()

    df = df.groupby(["user_id", "payee_id"], group_keys=False).apply(per_user_payee, include_groups=False)
    return df


def build_regime(obs_cfg: ObservableConfig, latent_cfg: LatentProcessConfig):
    rng = np.random.default_rng(obs_cfg.seed)

    latent_df, latent_cfg_used = generate_ground_truth(obs_cfg.n_rows, latent_cfg)
    is_candidate = latent_df["is_candidate"].values
    c = latent_df["C"].values
    label = latent_df["label"].values

    era = _assign_eras(obs_cfg.n_rows, rng)
    user_ids, payee_ids, mule_ids = _assign_users_and_payees(obs_cfg.n_rows, obs_cfg,
                                                              is_candidate, c, era, rng)
    timestamps = _assign_timestamps(obs_cfg.n_rows, obs_cfg, is_candidate, c, era, rng)
    amounts = _sample_amounts(is_candidate, c, user_ids, timestamps, obs_cfg, rng)

    context_df = _sample_context_features(obs_cfg.n_rows, is_candidate, c, rng)
    payee_side_df = _sample_payee_side_features(obs_cfg.n_rows, payee_ids, mule_ids,
                                                 is_candidate, c, rng)

    day_of_week = timestamps.dayofweek.values
    hour_of_day = timestamps.hour.values
    # correlated-but-non-causal shopping category: candidates skew slightly
    # toward "person_to_person"/"other" but with heavy overlap -> weak confound
    cat_weights_base = np.array([0.18, 0.16, 0.12, 0.14, 0.12, 0.16, 0.07, 0.05])
    cat_weights_cand = np.array([0.08, 0.08, 0.06, 0.30, 0.06, 0.10, 0.05, 0.27])
    shopping_category = []
    for cand, cc in zip(is_candidate, c):
        w = cat_weights_base * (1 - 0.5 * cc) + cat_weights_cand * (0.5 * cc) if cand else cat_weights_base
        w = w / w.sum()
        shopping_category.append(rng.choice(SHOPPING_CATEGORIES, p=w))
    instrument_type = rng.choice(INSTRUMENTS, size=obs_cfg.n_rows, p=[0.55, 0.40, 0.05])

    txn_id = [f"TXN{obs_cfg.regime_name.upper()}{i:07d}" for i in range(obs_cfg.n_rows)]
    vpa_pool = {}
    def vpa_for(pid):
        if pid not in vpa_pool:
            vpa_pool[pid] = f"{fake.user_name()}@{rng.choice(['okhdfc','oksbi','okicici','ybl','paytm'])}"
        return vpa_pool[pid]
    payee_vpa = [vpa_for(p) for p in payee_ids]

    base = pd.DataFrame({
        "txn_id": txn_id,
        "user_id": user_ids,
        "payee_id": payee_ids,
        "payee_vpa": payee_vpa,
        "timestamp": timestamps,
        "amount": amounts,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "shopping_category": shopping_category,
        "instrument_type": instrument_type,
        "era": era,  # generation-time fit/dev/test cohort bookkeeping --
                     # see train.py docstring for why splitting on this
                     # exact column (rather than re-deriving a row-count
                     # quantile) is what keeps the split exactly
                     # entity-disjoint instead of "disjoint up to a
                     # boundary-rounding mismatch". Not derived from C or
                     # the label; purely a partition id, analogous to a
                     # real system's onboarding-cohort/vintage field.
        "is_candidate": is_candidate,
        "C": c,
        "label": label,
        "raw_label_pre_noise": latent_df["raw_label_pre_noise"].values,
        "label_flipped_by_noise": latent_df["label_flipped_by_noise"].values,
    })
    base = pd.concat([base, context_df, payee_side_df], axis=1)

    base = _rolling_user_features(base)
    base = _rolling_payee_features(base)

    # fill first-appearance NaNs (no prior history) with sensible zeros
    fill_zero_cols = ["user_prior_txn_count_90d", "user_mean_amt_90d", "user_p95_amt_90d",
                       "user_max_amt_90d", "user_txn_count_last_1h", "user_txn_count_last_24h",
                       "payee_velocity_24h", "user_payee_txn_count_90d"]
    for col in fill_zero_cols:
        base[col] = base[col].fillna(0.0)
    base["time_since_last_txn_hours"] = base["time_since_last_txn_hours"].fillna(24 * 30)
    # a brand-new payee relationship: novelty is "large" (sentinel), not 0
    base["payee_novelty_days"] = base["payee_novelty_days"].fillna(9999.0)

    diag = diagnostics(latent_df, latent_cfg_used)
    diag["regime"] = obs_cfg.regime_name
    diag["mule_payee_share_of_txns"] = float(base["_is_mule_payee"].mean())

    observed_cols = [c for c in base.columns if c not in (
        "is_candidate", "C", "raw_label_pre_noise", "label_flipped_by_noise", "_is_mule_payee")]
    observed = base[observed_cols].sort_values("timestamp").reset_index(drop=True)
    full_latent = base.sort_values("timestamp").reset_index(drop=True)

    return observed, full_latent, diag


def main():
    import json, os
    os.makedirs("data", exist_ok=True)

    print("=" * 70)
    print("Generating REGIME A (in-process train/dev/test pool)")
    print("=" * 70)
    obs_a, full_a, diag_a = build_regime(REGIME_A, LATENT_CFG_A)
    obs_a.to_csv("data/regime_a_observed.csv", index=False)
    full_a.to_csv("data/regime_a_full_latent.csv", index=False)
    print(json.dumps(diag_a, indent=2, default=str))

    print("\n" + "=" * 70)
    print("Generating REGIME B (cross-process holdout -- scored ONCE, at the end)")
    print("=" * 70)
    obs_b, full_b, diag_b = build_regime(REGIME_B_CROSS_PROCESS, LATENT_CFG_B)
    obs_b.to_csv("data/regime_b_cross_process_observed.csv", index=False)
    full_b.to_csv("data/regime_b_cross_process_full_latent.csv", index=False)
    print(json.dumps(diag_b, indent=2, default=str))

    print("\nWrote:")
    print("  data/regime_a_observed.csv                 (features.py / train.py may read this)")
    print("  data/regime_a_full_latent.csv               (diagnostics ONLY -- has C, is_candidate)")
    print("  data/regime_b_cross_process_observed.csv    (features.py / train.py may read this)")
    print("  data/regime_b_cross_process_full_latent.csv (diagnostics ONLY -- has C, is_candidate)")


if __name__ == "__main__":
    main()
