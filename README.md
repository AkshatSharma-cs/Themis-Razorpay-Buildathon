# Sentinel — synthetic data + baseline model for "authorized but deceived" UPI scam detection

Razorpay AI Buildathon 2026, Track 2 (AI Risk Manager). Solo build, zero
budget, CPU-only. Libraries used: numpy, pandas, scipy (transitively via
sklearn), faker, scikit-learn, LightGBM, shap — all FOSS, no paid API or
service anywhere in this repo.

## Files, in the order you should read them

1. **`latent_process.py`** — the ONLY file that knows ground truth. Owns
   the latent coercion intensity `C` and the final noisy label. Nothing
   else in the repo imports this module or is allowed to see `C`.
2. **`generate.py`** — builds the observable UPI transaction dataset,
   conditioned on (but never containing) `C`. Writes two files per
   regime: a `*_full_latent.csv` (diagnostics only, has `C`) and a
   `*_observed.csv` (the only file downstream code may read). Also
   builds the cross-process holdout (a different amount/seasonality/
   scam-mix regime), and partitions users/payees into **non-overlapping
   fit/dev/test cohorts at generation time** — see "why the split
   worked" below.
3. **`features.py`** — the model-facing feature matrix, from
   `*_observed.csv` only. Explicitly drops all ID columns. Each feature
   group has a one-line comment on why it matters for *this* fraud
   class specifically.
4. **`train.py`** — splits, trains (LightGBM + logistic regression +
   rules baseline), calibrates (Platt scaling), thresholds by a cost
   ledger, runs SHAP, and scores the cross-process holdout exactly once.

## Running it

```bash
pip install --break-system-packages numpy pandas scipy faker scikit-learn lightgbm shap
python3 generate.py     # writes data/*.csv
python3 features.py     # smoke-test of the feature matrix (optional)
python3 train.py        # writes artifacts/sentinel_model.joblib + artifacts/metrics.json
```

## Design decisions made specifically to keep the eval honest

- **The capped logistic link** (`0.5 + 0.35·sigmoid(k·(C-0.5))` in
  `latent_process.py`): pins the maximum possible P(scam | C) at 0.85
  and the floor around 0.52, so no model — however good — can look
  "perfect" on this data. This is what keeps within-process AUROC well
  under 1.0 regardless of model quality.
- **Class-conditional (not symmetric) label noise**: a naive "flip 4-8%
  of ALL labels" is a well-known trap on a ~3% base rate — 6% of a
  97%-negative population is itself a bigger number of new fake
  positives than the entire true-positive class. We flip raw positives
  → negative at ~20% (under-reporting / failed adjudication) and raw
  negatives → positive at ~1-2% (false fraud flags), which keeps the
  *aggregate* noise auditable via `latent_process.diagnostics()` without
  destroying the minority class.
- **Amount is keyed off `C`/`is_candidate`, not off the final label.**
  We initially built it off `label` directly — and got a within-process
  AUROC of 0.99, because a noise-flipped label always came with (or
  without) a giveaway-huge amount attached, making amount a near-perfect
  proxy for the (noisy) label rather than a genuine, imperfect
  correlate of coercion. Rewriting it to escalate with *probability*
  rising in `C` fixed this — see the docstring in `generate._sample_amounts`.
- **Generation-time cohort partitioning for entity-disjointness.** We
  first tried "temporal cut, then discard any user/payee that straddles
  it" — this discarded ~98% of rows, because Zipf-popular "power users"
  span the whole timeline by construction. The fix: partition users and
  payees (including the mule pool) into non-overlapping fit/dev/test
  cohorts *before* generating any row, and give each cohort its own,
  non-overlapping, time-ordered window. `train.py` then splits on that
  same `era` label directly (not a re-derived quantile cut) — see the
  docstring in `temporal_entity_disjoint_split` for why re-deriving the
  cutoff by row-count quantile reintroduced a small amount of boundary
  leakage in development.
- **Distinct RNG seeds for the latent process vs. the observable
  generator.** We originally reused the same seed value for both. Two
  independently-constructed `np.random.default_rng` streams seeded
  identically produce the *same* underlying uniform sequence, and both
  pipelines' first operations were threshold-style draws
  (`is_candidate`, `era`) — so every `is_candidate=True` row was
  deterministically landing in the `fit` era, starving dev/test of
  positives almost entirely. Confirmed with a standalone repro; fixed
  by using different seeds.
- **Regularization is deliberately conservative** (shallow trees, capped
  `scale_pos_weight`, no early stopping on dev). With only ~100-150
  positives in `fit`, an unconstrained LightGBM reaches ~1.0 train AUROC
  by memorizing idiosyncratic feature combinations and generalizes worse
  than a plain logistic regression. We saw this directly: test AUROC
  dropped *below 0.5* (an unambiguous overfitting signature, not "no
  signal") before the hyperparameters were reined in.

## Self-audit: actual results vs. the target ranges

| Metric | Target (per spec) | Achieved | Verdict |
|---|---|---|---|
| Within-process AUROC | 0.85 – 0.92 | **0.810** | Slightly below |
| Cross-process AUROC | 0.75 – 0.85 | **0.680** | Below |
| Recall @ cost threshold | 0.55 – 0.75 | **0.406** | Below |
| FPR @ cost threshold | 0.01 – 0.03 | **0.007** | Below (more conservative) |
| Null-model AUROC | ~0.5 | **0.500** | On target |

**Honest read on the gap, not a cover story:** none of this traces to
leakage — the SHAP leakage check is clean (no ID-derived feature in the
top 10; the assertion that ID columns never reach the feature matrix
passes), and the entity-disjoint check reports zero user/payee overlap
across fit/dev/test. Two things are genuinely driving the shortfall:

1. **Small-N variance.** `test` and the cross-process holdout carry
   ~30 positives each. Re-running the identical pipeline across eight
   different LightGBM `random_state` values (holding everything else
   fixed) swings within-process AUROC between 0.80 and 0.84 and
   cross-process between 0.67 and 0.70 — the reported numbers sit
   squarely inside that natural band, not at an unlucky tail.
2. **Deliberately conservative regularization** (see above) trades a bit
   of headline AUROC for a model that isn't just memorizing `fit`. The
   plain logistic-regression baseline actually edges out LightGBM here
   (0.840 vs. 0.810 within-process) — worth flagging to the panel as a
   genuine finding: at this data scale, the simpler linear baseline is
   competitive, and a production build should not assume the fancier
   model is automatically better without re-checking as more labeled
   data accumulates.

We chose to report this honestly rather than keep tuning
hyperparameters/cost-ledger constants until the topline numbers matched
the target band — doing that would have been fitting the model to the
eval's target ranges rather than to the data, which defeats the purpose
of having target ranges at all. A larger labeled dataset (the single
lever with the most headroom here) would be the first thing to change
before re-reading these numbers.

## Cost ledger assumptions (documented, not derived)

- **FP cost = ₹150 flat** per false positive: support-desk touch +
  customer time + goodwill erosion from a wrongly-delayed legitimate
  payment. Flat, not amount-scaled, because the friction of an extra
  verification step doesn't really change with the payment size.
- **FN cost = scam amount × (1 − 0.12)**: UPI social-engineering scam
  recovery is low and slow once funds clear a mule chain; 12% is a
  conservative assumed eventual-recovery probability.

Both are named constants at the top of `train.py` (`FP_COST_RUPEES`,
`ASSUMED_RECOVERY_PROB`) — change them there if the panel wants to see
sensitivity to different assumptions; the threshold sweep and the
selected operating point will update automatically.

## Known limitations / what we'd fix with more time or more data

- `payee_velocity_24h` and `user_payee_txn_count_90d` end up with low
  SHAP importance — the "favorite payee" recurrence model that makes
  `payee_novelty_days` informative could be extended to give velocity a
  cleaner signal too (right now most rows still have velocity=0 in a
  24h window given payee pool size).
- `hour_of_day` is currently the single largest SHAP feature — larger
  than intended relative to the session-context group (call/screen/OTP
  signals), suggesting the odd-hour mixing probability in
  `generate._assign_timestamps` is a bit stronger than the other
  observable correlates of `C`. Worth re-balancing in a v2 so the
  session-context features (the ones that are actually specific to this
  fraud class) carry more of the model's decision.
- All numbers above are from a single generation seed. Reporting a
  mean ± range across a handful of seeds (shown informally in this
  README, not automated in code) would be a natural next addition for a
  more rigorous panel presentation.
