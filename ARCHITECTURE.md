# Themis — Architecture

**Track:** Razorpay AI Buildathon 2026, Track 2 (authorized-but-deceived UPI scam detection)
**Scope:** defense-only. Themis never blocks a payment, reverses funds, initiates a
chargeback, touches a payee's account, or acts against any party other than showing the
payer a warning. That line is enforced in code (`DefenseAction` in `serve.py`), not just
described here.

Every component below exists to solve a specific constraint of this build: zero budget,
solo developer, a track disqualification line that must never be crossed, and a live demo
that has to survive a flaky free-tier LLM call. Nothing here is included because it "sounds
impressive."

---

## 1. Service layer — FastAPI (`serve.py`)

**What it does:** exposes `/score`, `/decision`, `/release/*`, `/audit/*`, `/health`.

**Why FastAPI over Gradio:** the deliverable is a decision *engine*, not a single UI screen.
Judges, a hypothetical bank backend, and a future frontend all need typed, independently
callable JSON endpoints — that's a REST API, not a Gradio app. FastAPI's free automatic
`/docs` (OpenAPI/Swagger) doubles as a demo surface with zero extra work: you can literally
run the whole score → decision → release flow from a browser during judging. If you later
want a visual demo, build it as a thin client (Gradio, static HTML, whatever) that *calls*
this API — don't fuse the two, or the API stops being independently testable.

**Why the model loading has a dummy fallback:** the real LightGBM model and `metrics.json`
come from a separate ML workstream. The service is written so it runs and is fully testable
*before* those artifacts exist — `ModelBundle` falls back to a deterministic dummy scorer
and logs `is_dummy_model: true` in `/health` so nobody mistakes a placeholder for the real
system. Drop in the real `model.joblib` and `metrics.json` and nothing else changes.

**Why SHAP (`TreeExplainer`) for the top-3 features:** LightGBM is a tree ensemble, and
`TreeExplainer` gives exact, fast, model-specific attributions — no sampling, no
approximation error, and no per-request retraining. This is what makes the narration
(section 3) and the audit trail *explainable*, which regulated fintech reviewers will
expect.

---

## 2. The hard invariant: `DefenseAction`

This is the single most important piece of code in the repo, because getting it wrong is
an instant disqualification for this track.

```python
@dataclass(frozen=True)
class DefenseAction:
    action_type: Literal["cooling_off", "advisory_only", "none"]
    ...
    def __post_init__(self):
        if self.action_type not in _ALLOWED_ACTION_TYPES: raise ValueError(...)
        # also scans reason/verification text for forbidden terms:
        # "block", "reverse", "chargeback", "freeze", "payee_account", ...
```

**Why a hard invariant instead of a code-review convention or a comment:**

- A comment saying "never block" is advisory and can be violated by a future edit (by you,
  a teammate, or an LLM-assisted refactor) without anything failing. A dataclass whose
  `__post_init__` *raises* on construction is checked on every single instantiation,
  everywhere in the codebase, forever — there is no code path that can produce an illegal
  action and have it silently ship.
- There are exactly three classmethod constructors (`cooling_off`, `advisory_only`, `none`).
  There is no "escape hatch" constructor for a fourth action type. Anyone adding a new
  action type has to consciously edit this file and its validator, which makes the change
  visible in code review/diff rather than buried in a business-logic branch three files
  away.
- The forbidden-term scan is a second, independent check on the *text* of the reason —
  so even a correctly-typed `cooling_off` action can't accidentally describe itself as
  blocking or reversing something in its human-readable explanation.

**Escape path (`/release/request-otp`, `/release/confirm`):** a cooling-off is a *delay*,
not a lock — the payer can always end it immediately by re-authenticating with a fresh OTP
and explicitly confirming "I made this payment." This is the direct counterweight to giving
the system any power to slow a payment down at all: the system may never be more confident
than the payer about the payer's own intent.

**Daily cap (`COOLING_OFF_DAILY_CAP`, default 3):** counted from the audit log itself
(`count_events_today`), not a separate mutable counter, so the cap can't be reset or
tampered with independently of the tamper-evident trail. Once a payer hits the cap for the
day, further high-risk transactions get `advisory_only` (a non-blocking notice) instead of
another delay — Themis can slow someone down a bounded number of times, never trap them
in an unbounded loop of delays.

---

## 3. LLM narration (`narration.py`) — decision-support only, never the decision-maker

**Why the LLM never decides anything:** `/decision` computes the probability, compares it
to the cost-optimal threshold from `metrics.json`, and constructs the final `DefenseAction`
— all of that happens and is already final *before* `get_narration()` is ever called. The
LLM call can only change the *wording* of the explanation shown to the user; it cannot
change whether a cooling-off is issued, its duration, or the threshold. This mirrors how
regulated-decision systems (credit, fraud, medical triage) are expected to work in
practice: the auditable, deterministic model makes the call, and any generative layer is
strictly cosmetic/explanatory. It also completely derisks the demo — a Gemini or Groq
outage during judging cannot break the actual product.

**Why Gemini primary, Groq fallback, template last:**
- **Gemini (AI Studio, `gemini-2.0-flash`):** genuinely free forever, no card, generous
  request quota, fast enough for a live per-request explanation.
- **Groq:** also free/no-card, serves open-weight models (Llama) at very low latency —
  used only if Gemini errors or 429s, so the two failure modes (different vendors, different
  infra) are unlikely to fail together.
- **Deterministic template (`deterministic_template`):** built directly from the same
  top-3 SHAP features every other layer uses, with zero network dependency. This is framed
  as a **robustness feature**, not a degraded mode — it's arguably *more* trustworthy for an
  audit trail than an LLM sentence, because it's 100% reproducible from the stored inputs.
  Every call goes through a hard per-provider timeout (`NARRATION_TIMEOUT_S`, default 4s)
  via a thread pool, so a hung request can't stall the response — worst case, the total
  narration step costs ~2×timeout before falling through to the template.

---

## 4. Audit storage — SQLite + hash chain (`audit_db.py`)

**Why SQLite instead of a hosted DB (Supabase/Postgres/Mongo Atlas free tier):**
- **Cost/ops:** zero setup, zero external dependency, no risk of hitting a free-tier gotcha
  mid-demo. Supabase's free tier specifically auto-pauses a project after ~7 days of
  inactivity — exactly the kind of thing that silently breaks a demo you haven't touched
  in a week. SQLite has no such failure mode; the file just sits there.
- **It's a legible security property, not just "cheaper":** an append-only file with a hash
  chain is something a judge can understand and verify *by watching it happen* — run
  `verify_chain()`, then hand-edit one row with a raw SQL `UPDATE`, then run `verify_chain()`
  again and watch it flip to `ok: false` and point at the exact row. That's a concrete,
  demoable claim ("here's how I'd catch someone editing the audit log after the fact") that
  a hosted DB with row-level ACLs doesn't give you for free — ACLs stop *future* writes,
  they don't prove *past* rows weren't altered by someone with DB access (an insider, a
  compromised admin account, a support engineer with a console).

**How the chain works:** each row stores
`row_hash = SHA256(prev_hash | event_id | tx_id | payer_vpa | event_type | canonical_json(payload) | created_at)`.
Changing any field in any historical row changes that row's recomputed hash, which no
longer matches the *next* row's stored `prev_hash` — the break is immediately localized to
a specific row ID by `verify_chain()`, which is exactly what you want to show live.

**What's deliberately NOT in the chain:** OTPs (`_otp_store` in `serve.py`) are ephemeral
secrets, not audit facts — only the *event* that a valid OTP + confirmation occurred
(`COOLING_OFF_RELEASED`) is written to the chain. Mixing secrets into an audit log you want
to eventually show or export is a liability with no benefit.

---

## 5. Deployment — free tier only, no card anywhere

**Primary: Hugging Face Spaces, CPU Basic tier.** Free forever, no card required, 2 vCPU /
16 GB RAM — comfortably enough for FastAPI + LightGBM + SHAP on CPU. Use the Docker SDK
(a `Dockerfile` that runs `uvicorn serve:app --host 0.0.0.0 --port 7860`) so you have full
control over dependencies (`shap`, `lightgbm`, `google-generativeai`, `groq`) rather than
fighting the Gradio/Streamlit SDK templates.

> **Demo reminder:** CPU Basic Spaces sleep after ~48h of no traffic. **Before recording or
> presenting, load the Space URL yourself and wait for it to finish waking up** (can take
> 30–60s from cold) — do this a few minutes before you actually need it live, not in front
> of the judges.

**Optional: static landing/architecture page — Cloudflare Pages + Workers.** Free, no card,
generous request limits. Use this only if you want a separate marketing/architecture page
distinct from the live app (e.g. a one-pager linking to the HF Space, this ARCHITECTURE.md
rendered nicely, and a diagram). It should not host any part of the actual scoring logic —
keep the decision engine in one place (the HF Space) so the tamper-evident audit log has a
single source of truth.

**Explicitly rejected, and why:**
- *Supabase (hosted Postgres) free tier* — auto-pauses after 7 days idle; wrong tool anyway
  since the hash chain is the differentiator, not relational features we don't need.
- *Any "free trial requiring a card"* (Railway, Render's card-gated tiers, most cloud
  provider "free credits," MongoDB Atlas in some regions) — flagged and avoided outright.
  If you hit a wall on HF Spaces (e.g. need GPU), the FOSS alternative is to keep everything
  CPU-only — LightGBM + SHAP don't need a GPU — rather than reaching for a paid tier.
- *A managed SMS/OTP provider for `/release/request-otp`* — every free tier we found
  requires a card or business KYC. The demo returns the OTP directly in the API response,
  clearly labeled `otp_demo_only`, with a one-line note on swapping in a real provider
  (MSG91 / Twilio Verify) later — the rest of the release flow doesn't change.

---

## Summary — one line per component

| Component | Why it exists |
|---|---|
| FastAPI service | Independently callable, typed endpoints + free OpenAPI docs for judging |
| `DefenseAction` invariant | Code-level, not comment-level, enforcement of the track's disqualification line |
| Cooling-off + escape path + daily cap | Bounded, reversible-by-the-payer friction — never a lock, never unbounded |
| SHAP top-3 | Exact, model-native explanations feeding both narration and audit |
| Gemini → Groq → template narration | Free, fast, and — critically — never on the critical path of the actual decision |
| SQLite + hash chain | Zero-cost, zero-ops, and a demoable tamper-evidence property a hosted DB's ACLs don't give you |
| HF Spaces (CPU Basic) | Free-forever hosting sized for LightGBM + SHAP on CPU |
| Cloudflare Pages (optional) | Separate static surface, kept out of the decision engine's single source of truth |
