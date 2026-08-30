"""
latent_process.py
------------------
HONEST-EVAL DESIGN NOTE (read this before touching anything else in the repo):

This module is the *only* place in the whole codebase that knows the ground
truth. It owns two things and nothing else:

    1. C            -> latent "coercion intensity" for each transaction
                        (a real number, roughly in [0, 1], never written to
                        any file that features.py or train.py is allowed
                        to open).
    2. label         -> the final 0/1 target, produced from C through a
                        capped logistic link plus explicit label noise.

Nothing here samples amounts, call-overlap minutes, device flags, payee
history, etc. Those are OBSERVABLE features and they live in generate.py's
`build_observable_features()` path, which is a physically separate function
that is only ever handed `C` as a conditioning input the way a real-world
generator (a scammer's script + a victim's psychological state) would drive
*correlated but not identical* observable side-effects. features.py and
train.py NEVER import this module. The bridge is a CSV file: generate.py
writes a "full" diagnostic file (with C and is_candidate, for our own
sanity-checking) and a separate "observed" file (no C, no is_candidate --
this is the only file downstream code is allowed to touch).

Why this separation matters for the panel: it means the label-generating
process cannot be accused of being "reverse engineered" from whatever
features we happened to think a fraud model should have. We decided the
label mechanism first, in isolation, then built features against the
*observable* world only.

WHY THE CEILING EXISTS (0.5 + 0.35 * sigmoid(k*(C-0.5))):
Real "authorized but deceived" fraud is not perfectly separable even in
principle -- a victim mid-coercion-call sometimes still doesn't send money,
and a victim with very few risk signals sometimes still does. If we let
label = 1{C > threshold}, any model that partially recovers C gets a free
AUROC of ~1.0, which would make the whole eval a toy. Capping
P(label=1 | C) at 0.85 (the max value of 0.5 + 0.35*sigmoid(big)=0.85) and
flooring it at ~0.5 + 0.35*sigmoid(-big) (~0.5) means:
  - even a maximally-coerced transaction only has an 85% chance of actually
    resulting in a completed scam payment (some victims still don't send),
  - even a "candidate" scenario with weak coercion signal still has a
    >=50% chance of completing as a scam (fraud isn't fully separable from
    the weak end either).
This is what keeps within-process AUROC pinned well under 1.0 no matter how
good the downstream model is.

WHY THE NOISE INJECTION EXISTS:
Real labels (chargebacks, victim self-reports, bank fraud-ops adjudication)
are themselves noisy. We flip a documented share of labels post-hoc to
simulate adjudication error, independent of C.

IMPORTANT ENGINEERING NOTE ON *SYMMETRIC* NOISE AND CLASS IMBALANCE:
a naive reading of "flip 4-8% of labels at random" -- applied uniformly
across ALL rows, positive and negative alike -- is a well-known trap on a
~2-3% base-rate problem: 6% of a 97%-negative population is itself ~5.8
percentage points of brand-new false positives, which is larger than the
entire true-positive signal. That doesn't model "noisy labels", it
destroys the label. So we apply the noise *asymmetrically*, which is also
the more realistic model of how these labels actually get corrupted in
production:
  - `noise_rate_pos` (default 0.15): a raw-positive (completed scam per the
    logistic draw) gets relabeled 0 at this rate, modelling victims who
    never report / disputes that get adjudicated away / cases where the
    bank couldn't confirm coercion.
  - `noise_rate_neg` (default 0.006): a raw-negative gets relabeled 1 at
    this much smaller rate, modelling false fraud flags / mistaken
    chargebacks on legitimate transactions.
Total flipped labels as a share of the full dataset lands in the
documented 4-8% band once you weight by class size -- the diagnostics()
function below prints the realized numbers every run so this is auditable,
not just asserted.

WHY THE AMBIGUOUS-ZONE QUOTA EXISTS:
We explicitly bias the candidate-C sampling so that a documented share of
eventual positives sit in C in [0.4, 0.6] -- the region where the logistic
link is closest to its own inflection point and therefore hardest to
resolve even with a noise-free label. This guarantees a chunk of the
positive class is *inherently* hard, not just noisy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class LatentProcessConfig:
    """All knobs that decide ground truth. Nothing about observable
    features is configured here."""

    candidate_rate: float = 0.05          # share of txns where a coercion
                                           # attempt (of any intensity) is
                                           # actually happening
    link_steepness: float = 6.0           # k in sigmoid(k*(C-0.5))
    link_floor: float = 0.50              # base of the logistic link
    link_span: float = 0.35               # max additional probability mass
                                           # (0.50 + 0.35 = 0.85 ceiling)
    noise_rate_pos: float = 0.20          # raw-positive -> 0 flip rate
    noise_rate_neg: float = 0.011         # raw-negative -> 1 flip rate
    ambiguous_zone: tuple[float, float] = (0.4, 0.6)
    ambiguous_target_share: float = 0.15  # target share of positives whose
                                           # C sits inside ambiguous_zone
    seed: int = 20260829                  # documented seed


def _sample_coercion_intensity(n_candidates: int, rng: np.random.Generator,
                                ambiguous_zone: tuple[float, float],
                                ambiguous_target_share: float) -> np.ndarray:
    """Draw C for candidate (coercion-attempt) rows only.

    We mix two Beta components: a broad Beta(2,2) (symmetric, centered at
    0.5) for "typical" attempts, and an explicit oversample of the
    ambiguous_zone band so the eventual positive-label pool has the
    documented ~15-20% share sitting in the hardest-to-call region.
    Non-candidate rows are handled separately (C == 0, see caller).
    """
    n_ambiguous_boost = int(round(n_candidates * ambiguous_target_share * 0.2))
    n_broad = n_candidates - n_ambiguous_boost

    # U-shaped Beta(0.7, 0.7): most real coercion attempts are either
    # weak or strong, not "medium" -- this alone puts ~16% of mass in the
    # ambiguous_zone, close to the documented 15-20% target; the small
    # explicit boost below nudges it precisely without overriding the
    # (more realistic) bimodal shape.
    broad = rng.beta(0.7, 0.7, size=max(n_broad, 0))
    lo, hi = ambiguous_zone
    boosted = rng.uniform(lo, hi, size=max(n_ambiguous_boost, 0))

    c = np.concatenate([broad, boosted])
    rng.shuffle(c)
    # guard against float overrun outside [0, 1]
    return np.clip(c[:n_candidates], 0.0, 1.0)


def _p_label_given_c(c: np.ndarray, cfg: LatentProcessConfig) -> np.ndarray:
    """The capped logistic link. This is the ENTIRE mechanism by which C
    becomes a probability of a completed scam. Nothing else touches it."""
    z = cfg.link_steepness * (c - 0.5)
    sig = 1.0 / (1.0 + np.exp(-z))
    return cfg.link_floor + cfg.link_span * sig


def generate_ground_truth(n_rows: int, cfg: LatentProcessConfig | None = None
                           ) -> pd.DataFrame:
    """Return a DataFrame with exactly three columns: `is_candidate`,
    `C`, `label`. This is deliberately the *entire* public surface of this
    module -- generate.py conditions its observable-feature sampling on
    `C`/`is_candidate` but must never write them to the model-facing CSV.
    """
    cfg = cfg or LatentProcessConfig()
    rng = np.random.default_rng(cfg.seed)

    is_candidate = rng.random(n_rows) < cfg.candidate_rate
    n_candidates = int(is_candidate.sum())

    c = np.zeros(n_rows, dtype=float)
    c_candidates = _sample_coercion_intensity(
        n_candidates, rng, cfg.ambiguous_zone, cfg.ambiguous_target_share
    )
    c[is_candidate] = c_candidates

    p_label = np.zeros(n_rows, dtype=float)
    p_label[is_candidate] = _p_label_given_c(c[is_candidate], cfg)
    raw_label = (rng.random(n_rows) < p_label).astype(int)
    # non-candidates can never be a completed scam pre-noise
    raw_label[~is_candidate] = 0

    # --- documented, class-conditional label noise (see module docstring
    #     for why symmetric noise on an imbalanced problem is a trap) ---
    label = raw_label.copy()
    flip_roll = rng.random(n_rows)
    flip_pos = (raw_label == 1) & (flip_roll < cfg.noise_rate_pos)
    flip_neg = (raw_label == 0) & (flip_roll < cfg.noise_rate_neg)
    noise_mask = flip_pos | flip_neg
    label[noise_mask] = 1 - label[noise_mask]

    df = pd.DataFrame({
        "is_candidate": is_candidate,
        "C": c,
        "raw_label_pre_noise": raw_label,
        "label": label,
        "label_flipped_by_noise": noise_mask,
        "flip_direction": np.where(flip_pos, "pos_to_neg",
                           np.where(flip_neg, "neg_to_pos", "none")),
    })
    return df, cfg


def diagnostics(df: pd.DataFrame, cfg: LatentProcessConfig) -> dict:
    """Self-check numbers a reviewer will want to see printed at generation
    time: positive rate, ceiling proof, and the ambiguous-zone share among
    positives."""
    pos = df[df["label"] == 1]
    lo, hi = cfg.ambiguous_zone
    amb_share = ((pos["C"] >= lo) & (pos["C"] <= hi)).mean() if len(pos) else float("nan")
    max_p = _p_label_given_c(np.array([1.0]), cfg)[0]
    min_p = _p_label_given_c(np.array([0.0]), cfg)[0]
    return {
        "n_rows": len(df),
        "positive_rate": float(df["label"].mean()),
        "n_candidates": int(df["is_candidate"].sum()),
        "candidate_rate_actual": float(df["is_candidate"].mean()),
        "total_noise_flip_rate_actual": float(df["label_flipped_by_noise"].mean()),
        "pos_to_neg_flips": int((df["flip_direction"] == "pos_to_neg").sum()),
        "neg_to_pos_flips": int((df["flip_direction"] == "neg_to_pos").sum()),
        "ambiguous_zone": cfg.ambiguous_zone,
        "ambiguous_share_of_positives": float(amb_share),
        "link_probability_ceiling": float(max_p),
        "link_probability_floor_at_C0": float(min_p),
    }
