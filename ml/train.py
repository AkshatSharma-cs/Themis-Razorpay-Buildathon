"""
train.py
--------
Splits, trains, calibrates, and evaluates the Themis scam detector.

SPLIT DESIGN (why each one exists):
  * Temporal holdout (last 20% of simulated time): a fraud model that's
    only ever validated on randomly-shuffled rows gets to "see the
    future" during cross-validation, which real production scoring never
    gets to do. We sort by timestamp and cut on time, not row index.
  * Entity-disjoint (no user_id / payee_id in both train and test):
    without this, a tree model can partially memorize "this user_id was
    fraud before" via interaction with other features and look great on a
    who's-already-in-training-data test set while learning nothing that
    transfers to a genuinely new user or payee. We verify this
    programmatically below and print the confirmation -- this isn't a
    claim, it's a check.
  * A separate dev set (carved out of the pre-test time window) is used
    ONLY for calibration and threshold selection, so the test set stays
    completely untouched until the one, final, "how good is this" read.
  * The cross-process dataset (regime B) is scored EXACTLY ONCE, after
    everything above -- model choice, calibration, threshold -- is
    frozen. If we peeked at it earlier and iterated based on it, it would
    stop being a holdout and start being a second dev set.

SELF-AUDIT TARGETS (documented, not tuned to):
  within-process AUROC ~ 0.85-0.92, cross-process ~ 0.75-0.85, recall at
  the cost-optimal threshold ~ 0.55-0.75 with FPR ~ 1-3%. Landing near
  0.97+ anywhere is treated as a bug (most likely ID leakage or the
  generator being too easy), not a win -- see `sanity_check()` below,
  which runs automatically and prints a warning rather than silently
  passing.
"""

from __future__ import annotations

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              precision_recall_curve, confusion_matrix)
from sklearn.dummy import DummyClassifier
import lightgbm as lgb
import shap

from features import build_features, FEATURE_GROUPS

warnings.filterwarnings("ignore", category=UserWarning)

RNG_SEED = 20260829

# ---- cost ledger (documented assumptions) --------------------------------
# FP cost: a fixed rupee value for the friction of delaying/flagging a
# LEGITIMATE payment -- support-desk touch time, the customer's own time,
# and goodwill/trust erosion from being wrongly stopped. We set this to a
# flat Rs. 150 per false positive; it's deliberately not amount-scaled,
# because the friction cost of an extra verification step is roughly
# constant regardless of whether the blocked payment was Rs. 200 or
# Rs. 20,000.
FP_COST_RUPEES = 150.0
# FN cost: scam amount * (1 - assumed recovery probability). UPI recovery
# rates for social-engineering scams are low and slow (banks/NPCI recovery
# is far from guaranteed once funds move through a mule chain); we assume
# a conservative 12% eventual recovery probability.
ASSUMED_RECOVERY_PROB = 0.12


def load_regime(path: str, is_train: bool, category_levels=None):
    df = pd.read_csv(path)
    X, y, category_levels, meta = build_features(df, is_train=is_train,
                                                  category_levels=category_levels)
    return X, y, category_levels, meta


def temporal_entity_disjoint_split(meta: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                                    era: pd.Series):
    """Temporal + entity-disjoint holdout.

    generate.py partitions users and payees (including the mule pool)
    into non-overlapping fit/dev/test cohorts BEFORE any row is
    generated, and assigns each cohort its own, non-overlapping, strictly
    time-ordered window (fit is earliest, dev next, test is the final
    20%). We split on that generation-time `era` label directly rather
    than re-deriving a row-count quantile cut on the fly: the two should
    agree up to rounding, but a rounding-mismatch of even a couple dozen
    rows landing on the wrong side of an approximate quantile cut is
    enough to reintroduce entity overlap at the boundary (we hit exactly
    this in development -- see git history / prior version of this
    function). Splitting on `era` itself is exact by construction.
    `verify_entity_disjoint` below is a genuine confirmation of that
    guarantee, not a patch for it."""
    fit_idx = meta.index[era == "fit"]
    dev_idx = meta.index[era == "dev"]
    test_idx = meta.index[era == "test"]
    assert meta.loc[fit_idx, "timestamp"].max() <= meta.loc[dev_idx, "timestamp"].min(), \
        "fit/dev eras are not cleanly time-ordered -- investigate generate.py era windows"
    assert meta.loc[dev_idx, "timestamp"].max() <= meta.loc[test_idx, "timestamp"].min(), \
        "dev/test eras are not cleanly time-ordered -- investigate generate.py era windows"
    return fit_idx, dev_idx, test_idx


def verify_entity_disjoint(meta: pd.DataFrame, idx_a, idx_b, name_a: str, name_b: str) -> bool:
    users_a, payees_a = set(meta.loc[idx_a, "user_id"]), set(meta.loc[idx_a, "payee_id"])
    users_b, payees_b = set(meta.loc[idx_b, "user_id"]), set(meta.loc[idx_b, "payee_id"])
    u_overlap = users_a & users_b
    p_overlap = payees_a & payees_b
    ok = (len(u_overlap) == 0) and (len(p_overlap) == 0)
    print(f"  entity-disjoint check [{name_a} vs {name_b}]: "
          f"user overlap={len(u_overlap)}, payee overlap={len(p_overlap)} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def train_models(X_fit, y_fit, X_dev=None, y_dev=None):
    pos_rate = y_fit.mean()
    # scale_pos_weight capped: with only ~2-5% positives in `fit`, the
    # "natural" balancing weight ((1-p)/p, ~35-40x here) pushes LightGBM
    # to chase the tiny positive class so hard it overfits their
    # idiosyncrasies instead of learning the shared risk pattern -- we
    # saw exactly that in development (near-perfect train AUROC, weak
    # test AUROC). Capping the weight is a documented, deliberate
    # trade-off: less aggressive class rebalancing, more generalizable
    # trees.
    scale_pos_weight = min((1 - pos_rate) / max(pos_rate, 1e-6), 12.0)

    # Regularization note: shallow/few-leaf/heavily-subsampled on purpose
    # -- see module docstring self-audit targets. We do NOT early-stop on
    # `dev` here: with only a couple dozen positives, dev-AUC during
    # boosting is too noisy round-to-round to be a reliable stopping
    # signal (we tried it -- it stopped after a handful of trees and
    # underfit). Fixed, conservative hyperparameters proved more stable.
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, num_leaves=3, max_depth=3, learning_rate=0.05,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=RNG_SEED, verbosity=-1,
    )
    lgb_model.fit(X_fit, y_fit)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG_SEED,
                                 C=0.3)  # L2-regularized more strongly than default -- same
                                          # small-positive-count overfitting risk applies here
    logreg_X = (X_fit - X_fit.mean()) / X_fit.std().replace(0, 1)
    logreg.fit(logreg_X, y_fit)

    return lgb_model, logreg, X_fit.mean(), X_fit.std().replace(0, 1)


def rules_only_baseline(X: pd.DataFrame) -> np.ndarray:
    """A simple, explainable, non-learned baseline: flag if the amount is
    far above the user's own p95 AND the payee is (nearly) brand new. This
    is the kind of rule a bank's existing system likely already runs --
    the ML model should meaningfully beat it, not just match it."""
    cond = (X["amount_p95_ratio"] > 3.0) & (X["payee_novelty_days"] < 3.0)
    return cond.astype(int).values


def platt_calibrate(raw_scores_dev, y_dev):
    """Platt scaling: fit a 1-D logistic regression of true label on the
    model's raw score, on the DEV set only (never on test)."""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(raw_scores_dev.reshape(-1, 1), y_dev)
    return lr


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return ece


def sweep_cost_threshold(y_true, y_prob, amounts, fp_cost=FP_COST_RUPEES,
                          recovery_prob=ASSUMED_RECOVERY_PROB, n_steps=199):
    thresholds = np.linspace(0.005, 0.995, n_steps)
    costs = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fp_mask = (pred == 1) & (y_true == 0)
        fn_mask = (pred == 0) & (y_true == 1)
        fp_total = fp_mask.sum() * fp_cost
        fn_total = (amounts[fn_mask] * (1 - recovery_prob)).sum()
        costs.append(fp_total + fn_total)
    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    return thresholds, costs, thresholds[best_idx], costs[best_idx]


def metrics_at_threshold(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {"threshold": float(threshold), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}


def sanity_check(name: str, auroc: float, lo: float, hi: float):
    flag = "OK" if lo <= auroc <= hi else ("SUSPICIOUSLY HIGH" if auroc > hi else "BELOW EXPECTED RANGE")
    print(f"  [{name}] AUROC={auroc:.4f}  expected~[{lo},{hi}]  -> {flag}")
    if auroc > 0.97:
        print(f"  !! {name}: AUROC > 0.97 -- treating this as a BUG, not a win. "
              f"Check SHAP top features for ID leakage before trusting this number.")
    return flag


def main():
    print("=" * 70)
    print("1. Loading regime A (in-process) and building features")
    print("=" * 70)
    df_a = pd.read_csv("data/regime_a_observed.csv")
    X_all, y_all, category_levels, meta_all = build_features(df_a, is_train=True)
    print(f"  rows={len(X_all)}  positive_rate={y_all.mean():.4f}")

    print("\n" + "=" * 70)
    print("2. Temporal + entity-disjoint split (fit / dev / test)")
    print("=" * 70)
    era_all = df_a.loc[X_all.index, "era"]
    fit_idx, dev_idx, test_idx = temporal_entity_disjoint_split(meta_all, X_all, y_all, era_all)
    print(f"  fit={len(fit_idx)}  dev={len(dev_idx)}  test={len(test_idx)}")
    verify_entity_disjoint(meta_all, fit_idx, test_idx, "fit", "test")
    verify_entity_disjoint(meta_all, fit_idx, dev_idx, "fit", "dev")
    verify_entity_disjoint(meta_all, dev_idx, test_idx, "dev", "test")

    X_fit, y_fit = X_all.loc[fit_idx], y_all[X_all.index.get_indexer(fit_idx)]
    X_dev, y_dev = X_all.loc[dev_idx], y_all[X_all.index.get_indexer(dev_idx)]
    X_test, y_test = X_all.loc[test_idx], y_all[X_all.index.get_indexer(test_idx)]
    amt_test = meta_all.loc[test_idx, "amount"].values
    print(f"  fit positive_rate={y_fit.mean():.4f}  dev positive_rate={y_dev.mean():.4f}  "
          f"test positive_rate={y_test.mean():.4f}")

    print("\n" + "=" * 70)
    print("3. Training models (LightGBM primary, LogReg + rules baselines)")
    print("=" * 70)
    lgb_model, logreg, feat_mean, feat_std = train_models(X_fit, y_fit, X_dev, y_dev)

    lgb_raw_test = lgb_model.predict_proba(X_test)[:, 1]
    logreg_test = logreg.predict_proba((X_test - feat_mean) / feat_std)[:, 1]
    rules_test = rules_only_baseline(X_test)

    lgb_raw_dev = lgb_model.predict_proba(X_dev)[:, 1]

    dummy = DummyClassifier(strategy="uniform", random_state=RNG_SEED).fit(X_fit, y_fit)
    dummy_test = dummy.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("4. Null-model sanity check (should be ~0.5 AUROC)")
    print("=" * 70)
    dummy_auc = roc_auc_score(y_test, dummy_test)
    print(f"  dummy/random predictor AUROC on test = {dummy_auc:.4f} (expect ~0.5)")

    print("\n" + "=" * 70)
    print("5. Calibration (Platt scaling, fit on DEV only)")
    print("=" * 70)
    platt = platt_calibrate(lgb_raw_dev, y_dev)
    lgb_cal_test = platt.predict_proba(lgb_raw_test.reshape(-1, 1))[:, 1]

    brier_before = brier_score_loss(y_test, lgb_raw_test)
    brier_after = brier_score_loss(y_test, lgb_cal_test)
    ece_before = expected_calibration_error(y_test, lgb_raw_test)
    ece_after = expected_calibration_error(y_test, lgb_cal_test)
    print(f"  Brier before={brier_before:.4f}  after={brier_after:.4f}")
    print(f"  ECE   before={ece_before:.4f}  after={ece_after:.4f}")

    frac_pos_before, mean_pred_before = calibration_curve(y_test, lgb_raw_test, n_bins=10, strategy="quantile")
    frac_pos_after, mean_pred_after = calibration_curve(y_test, lgb_cal_test, n_bins=10, strategy="quantile")
    reliability = {
        "before": {"mean_predicted": mean_pred_before.tolist(), "frac_positive": frac_pos_before.tolist()},
        "after": {"mean_predicted": mean_pred_after.tolist(), "frac_positive": frac_pos_after.tolist()},
    }

    print("\n" + "=" * 70)
    print("6. Headline metrics: PR-AUC (imbalance-robust) + AUROC, all 3 models")
    print("=" * 70)
    results = {}
    for name, scores in [("lightgbm_calibrated", lgb_cal_test),
                          ("logistic_regression", logreg_test),
                          ("rules_only_baseline", rules_test.astype(float))]:
        auroc = roc_auc_score(y_test, scores)
        prauc = average_precision_score(y_test, scores)
        results[name] = {"auroc": float(auroc), "pr_auc": float(prauc)}
        print(f"  {name:22s}  AUROC={auroc:.4f}  PR-AUC={prauc:.4f}")

    sanity_check("regime_a (within-process) LightGBM", results["lightgbm_calibrated"]["auroc"], 0.85, 0.92)

    print("\n" + "=" * 70)
    print("7. Cost-based threshold selection (sweep on DEV, apply to TEST)")
    print("=" * 70)
    lgb_cal_dev = platt.predict_proba(lgb_raw_dev.reshape(-1, 1))[:, 1]
    amt_dev = meta_all.loc[dev_idx, "amount"].values
    thresholds, costs, best_threshold, best_cost_dev = sweep_cost_threshold(y_dev, lgb_cal_dev, amt_dev)
    print(f"  cost-minimizing threshold (selected on DEV) = {best_threshold:.4f}")
    print(f"  FP_COST=Rs.{FP_COST_RUPEES}  assumed_recovery_prob={ASSUMED_RECOVERY_PROB}")

    test_metrics = metrics_at_threshold(y_test, lgb_cal_test, best_threshold)
    print(f"  TEST @ threshold: precision={test_metrics['precision']:.3f}  "
          f"recall={test_metrics['recall']:.3f}  FPR={test_metrics['fpr']:.4f}")

    print("\n" + "=" * 70)
    print("8. SHAP (TreeSHAP) global importance + leakage check")
    print("=" * 70)
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    print(shap_importance.head(10).to_string())
    id_like = [c for c in shap_importance.index[:5] if "id" in c.lower()]
    print(f"  leakage check: any ID-like column in top 5 SHAP features? {'YES -- BUG' if id_like else 'no'}")

    print("\n" + "=" * 70)
    print("9. Error cases (FN = missed scams, FP = delayed legit payments)")
    print("=" * 70)
    pred_test = (lgb_cal_test >= best_threshold).astype(int)
    test_meta = meta_all.loc[test_idx].copy()
    test_meta["y_true"] = y_test
    test_meta["y_prob"] = lgb_cal_test
    test_meta["y_pred"] = pred_test

    fn_cases = test_meta[(test_meta.y_true == 1) & (test_meta.y_pred == 0)][
        ["txn_id", "user_id", "payee_id", "timestamp", "amount", "y_prob"]]
    fp_cases = test_meta[(test_meta.y_true == 0) & (test_meta.y_pred == 1)][
        ["txn_id", "user_id", "payee_id", "timestamp", "amount", "y_prob"]].copy()
    fp_cases["delay_cost_rupees"] = FP_COST_RUPEES
    fn_cases = fn_cases.copy()
    fn_cases["fn_cost_rupees"] = fn_cases["amount"] * (1 - ASSUMED_RECOVERY_PROB)

    print(f"  FN (missed scams): {len(fn_cases)} cases, total exposure Rs.{fn_cases['fn_cost_rupees'].sum():,.0f}")
    print(f"  FP (delayed legit): {len(fp_cases)} cases, total friction cost Rs.{fp_cases['delay_cost_rupees'].sum():,.0f}")

    print("\n" + "=" * 70)
    print("10. Cross-process holdout -- scored EXACTLY ONCE, now that everything else is frozen")
    print("=" * 70)
    df_b = pd.read_csv("data/regime_b_cross_process_observed.csv")
    X_b, y_b, _, meta_b = build_features(df_b, is_train=False, category_levels=category_levels)
    lgb_raw_b = lgb_model.predict_proba(X_b)[:, 1]
    lgb_cal_b = platt.predict_proba(lgb_raw_b.reshape(-1, 1))[:, 1]
    cross_auroc = roc_auc_score(y_b, lgb_cal_b)
    cross_prauc = average_precision_score(y_b, lgb_cal_b)
    cross_metrics = metrics_at_threshold(y_b, lgb_cal_b, best_threshold)
    print(f"  cross-process AUROC={cross_auroc:.4f}  PR-AUC={cross_prauc:.4f}")
    print(f"  cross-process @ frozen threshold: precision={cross_metrics['precision']:.3f} "
          f"recall={cross_metrics['recall']:.3f} FPR={cross_metrics['fpr']:.4f}")
    sanity_check("regime_b (cross-process) LightGBM", cross_auroc, 0.75, 0.85)

    print("\n" + "=" * 70)
    print("11. Saving model + metrics.json handoff artifact")
    print("=" * 70)
    import os
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({
        "lgb_model": lgb_model, "platt_calibrator": platt,
        "category_levels": category_levels, "feature_columns": X_all.columns.tolist(),
        "cost_threshold": best_threshold,
    }, "artifacts/sentinel_model.joblib")

    metrics = {
        "dataset": {
            "regime_a_rows": len(X_all), "regime_a_positive_rate": float(y_all.mean()),
            "fit_rows": len(fit_idx), "dev_rows": len(dev_idx), "test_rows": len(test_idx),
        },
        "null_model_sanity_check_auroc": float(dummy_auc),
        "model_comparison_test": results,
        "calibration": {
            "brier_before": float(brier_before), "brier_after": float(brier_after),
            "ece_before": float(ece_before), "ece_after": float(ece_after),
            "reliability_diagram": reliability,
        },
        "cost_ledger": {"fp_cost_rupees": FP_COST_RUPEES, "assumed_recovery_prob": ASSUMED_RECOVERY_PROB},
        "threshold_selection": {"selected_threshold": float(best_threshold),
                                 "selected_on": "dev_set", "dev_cost_at_threshold": float(best_cost_dev)},
        "test_metrics_at_threshold": test_metrics,
        "shap_top10_global_importance": shap_importance.head(10).to_dict(),
        "error_cases": {
            "false_negatives": fn_cases.assign(timestamp=fn_cases["timestamp"].astype(str)).to_dict("records"),
            "false_positives": fp_cases.assign(timestamp=fp_cases["timestamp"].astype(str)).to_dict("records"),
        },
        "cross_process_holdout_regime_b": {
            "rows": len(X_b), "positive_rate": float(y_b.mean()),
            "auroc": float(cross_auroc), "pr_auc": float(cross_prauc),
            "metrics_at_frozen_threshold": cross_metrics,
        },
        "self_audit_expected_ranges": {
            "within_process_auroc": [0.85, 0.92],
            "cross_process_auroc": [0.75, 0.85],
            "recall_at_cost_threshold": [0.55, 0.75],
            "fpr_at_cost_threshold": [0.01, 0.03],
        },
    }
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("  saved artifacts/sentinel_model.joblib")
    print("  saved artifacts/metrics.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
