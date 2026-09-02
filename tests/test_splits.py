"""
tests/test_splits.py

Confirms two of train.py's own headline claims, but as real pytest
assertions instead of printed console lines:

  1. the generation-time era partitioning (fit/dev/test) really is
     entity-disjoint on both user_id and payee_id -- mirrors train.py's
     verify_entity_disjoint() check.
  2. a null/random predictor scores ~0.5 AUROC on the test split -- mirrors
     train.py's own DummyClassifier sanity check.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import generate
import latent_process


@pytest.fixture(scope="module")
def small_split_regime():
    """A reduced-size regime that still lands enough positives in the test
    era for a meaningful (if noisy) AUROC computation, without paying for
    the full 5,000-row REGIME_A generation."""
    # n_payees / mule_payee_frac are chosen so the mule sub-pool (n_payees *
    # mule_payee_frac) comes out to ~9 payees: large enough that
    # generate.py's own three-way fit/dev/test partition of the mule pool
    # (65/15/20) doesn't round any era down to zero mules, which would
    # otherwise raise inside numpy's rng.choice on an empty pool.
    small_obs_cfg = replace(
        generate.REGIME_A,
        regime_name="regime_test_split",
        n_rows=1200,
        n_users=200,
        n_payees=150,
        mule_payee_frac=0.06,
    )
    # Higher candidate_rate than the real REGIME_A so a 1200-row dataset
    # still lands a reasonable number of positives per era -- purely a
    # test-speed accommodation; it does not change how the split itself
    # is constructed.
    small_latent_cfg = latent_process.LatentProcessConfig(candidate_rate=0.15, seed=777)
    observed, full_latent, diag = generate.build_regime(small_obs_cfg, small_latent_cfg)
    return observed


def test_eras_present_and_time_ordered(small_split_regime):
    observed = small_split_regime
    eras_seen = set(observed["era"].unique())
    assert eras_seen == {"fit", "dev", "test"}, (
        f"expected exactly the eras fit/dev/test, got {eras_seen}"
    )
    fit_max = observed.loc[observed.era == "fit", "timestamp"].max()
    dev_min = observed.loc[observed.era == "dev", "timestamp"].min()
    dev_max = observed.loc[observed.era == "dev", "timestamp"].max()
    test_min = observed.loc[observed.era == "test", "timestamp"].min()
    assert fit_max <= dev_min, (
        f"fit/dev eras are not time-ordered: fit ends {fit_max}, dev starts {dev_min}"
    )
    assert dev_max <= test_min, (
        f"dev/test eras are not time-ordered: dev ends {dev_max}, test starts {test_min}"
    )


@pytest.mark.parametrize("era_a,era_b", [
    ("fit", "dev"), ("fit", "test"), ("dev", "test"),
])
def test_user_and_payee_ids_are_entity_disjoint_across_eras(small_split_regime, era_a, era_b):
    observed = small_split_regime
    rows_a = observed[observed.era == era_a]
    rows_b = observed[observed.era == era_b]

    users_a, users_b = set(rows_a.user_id), set(rows_b.user_id)
    payees_a, payees_b = set(rows_a.payee_id), set(rows_b.payee_id)

    user_overlap = users_a & users_b
    payee_overlap = payees_a & payees_b

    assert not user_overlap, (
        f"user_id overlap between {era_a!r} and {era_b!r}: {user_overlap} -- a user "
        f"appearing in both breaks the entity-disjoint guarantee."
    )
    assert not payee_overlap, (
        f"payee_id overlap between {era_a!r} and {era_b!r}: {payee_overlap} -- a payee "
        f"appearing in both breaks the entity-disjoint guarantee."
    )


def test_every_row_has_exactly_one_era(small_split_regime):
    observed = small_split_regime
    assert observed["era"].isin(["fit", "dev", "test"]).all(), (
        "found a row with an era value outside {fit, dev, test}"
    )
    assert not observed["era"].isna().any(), "found a row with a missing era"


def test_null_model_auroc_is_approximately_half(small_split_regime):
    """A predictor that has learned nothing should score ~0.5 AUROC on
    average -- landing meaningfully away from 0.5 here would indicate a
    labeling bug or a test-split construction bug (e.g. only one class
    present), independent of any real model. We average several
    random-seed random-score draws to smooth out the small-N variance
    inherent to a few-hundred-row test split, mirroring the intent (not
    the exact mechanics) of train.py's DummyClassifier sanity check."""
    observed = small_split_regime
    test_labels = observed.loc[observed.era == "test", "label"].values

    assert test_labels.sum() > 0, (
        "test split has zero positive labels -- AUROC is undefined; increase "
        "candidate_rate or n_rows in the fixture."
    )
    assert test_labels.sum() < len(test_labels), (
        "test split has zero negative labels -- AUROC is undefined."
    )

    aurocs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        random_scores = rng.random(len(test_labels))
        aurocs.append(roc_auc_score(test_labels, random_scores))

    mean_auroc = float(np.mean(aurocs))
    assert 0.40 <= mean_auroc <= 0.60, (
        f"mean AUROC of a random/null predictor over 20 seeds was {mean_auroc:.4f}, "
        f"expected ~0.5 (within [0.40, 0.60]) -- a random predictor scoring "
        f"reliably away from 0.5 suggests a bug in the test split (e.g. label "
        f"leakage, or the test set not being representative)."
    )


def test_null_model_auroc_on_dev_split_is_also_approximately_half(small_split_regime):
    """Same sanity check applied to the dev split, so a bug specific to
    how `test` (vs `dev`) is carved out doesn't slip through unnoticed."""
    observed = small_split_regime
    dev_labels = observed.loc[observed.era == "dev", "label"].values

    if dev_labels.sum() == 0 or dev_labels.sum() == len(dev_labels):
        pytest.skip("dev split has only one class present for this seed/size -- "
                     "AUROC undefined, not a split-correctness failure by itself")

    aurocs = []
    for seed in range(20):
        rng = np.random.default_rng(seed + 1000)
        random_scores = rng.random(len(dev_labels))
        aurocs.append(roc_auc_score(dev_labels, random_scores))

    mean_auroc = float(np.mean(aurocs))
    assert 0.35 <= mean_auroc <= 0.65, (
        f"mean AUROC of a random/null predictor on the dev split over 20 seeds "
        f"was {mean_auroc:.4f}, expected ~0.5 (within [0.35, 0.65])"
    )
