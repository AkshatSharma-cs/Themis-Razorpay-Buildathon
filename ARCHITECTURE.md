# Themis — Architecture

**Track:** Razorpay AI Buildathon 2026, Track 2 (authorized-but-deceived UPI scam detection)
**Scope:** defense-only. Themis never blocks a payment, reverses funds, initiates a
chargeback, touches a payee's account, or acts against any party other than showing the
payer a warning. That line is enforced in code (`DefenseAction` in `serve.py`), not just
described here.

Each component addresses a project constraint: zero budget, CPU-only execution, the
defense-only track requirement, and resilience to free-tier LLM failures.

---

## 1. Service layer — FastAPI (`serve.py`)

**What it does:** exposes `/score`, `/decision`, `/release/*`, `/audit/*`, `/health`.

**Why FastAPI over Gradio:** the deliverable is a decision *engine*, not a single UI screen.
Judges, bank systems, and future clients require typed, independently callable JSON
endpoints. FastAPI also generates OpenAPI/Swagger documentation at `/docs`, allowing the
score, decision, release, and audit flows to be tested without coupling them to the UI.

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
- **Gemini (AI Studio, `gemini-2.0-flash`):** available on a free tier without a card,
  with sufficient quota for this demo.
- **Groq:** also free/no-card, serves open-weight models (Llama) at very low latency —
  used only if Gemini errors or 429s, so the two failure modes (different vendors, different
  infra) are unlikely to fail together.
- **Deterministic template (`deterministic_template`):** built directly from the same
  top-3 SHAP features every other layer uses, with zero network dependency. This is framed
  as a **robustness feature**: it is reproducible from the stored inputs and suitable for
  audit records.
  Every call goes through a hard per-provider timeout (`NARRATION_TIMEOUT_S`, default 4s)
  via a thread pool, so a hung request can't stall the response — worst case, the total
  narration step costs ~2×timeout before falling through to the template.

---

## 4. Audit storage — SQLite + hash chain (`audit_db.py`)

**Why SQLite instead of a hosted DB (Supabase/Postgres/Mongo Atlas free tier):**
- **Cost/ops:** zero setup, zero external dependency, no risk of hitting a free-tier gotcha
  mid-demo. Supabase's free tier specifically auto-pauses a project after ~7 days of
  inactivity. SQLite avoids that operational dependency for the demo.
- **Auditability:** an append-only file with a hash chain can be verified with
  `verify_chain()`. Modifying a historical row with SQL and rerunning verification changes
  the result to `ok: false` and identifies the broken row. Row-level ACLs restrict future
  writes but do not establish that historical rows were not altered.

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

## 5. Deployment — Render backend + Vercel frontend

**Backend: Render Free web service.** The FastAPI service runs on Render's free web-service
tier, with no card required for this project. The Docker image keeps the CPU-only runtime
and its dependencies (`shap`, `lightgbm`, `google-generativeai`, and `groq`) together. Render
starts Uvicorn on the platform-provided `PORT` value and exposes the live API at:
`https://themis-razorpay-buildathon.onrender.com`.

**Free-tier tradeoff:** Render spins a Free web service down after 15 minutes without
inbound traffic. The next HTTP request starts it again and the cold start takes about one
minute. The frontend health badge therefore distinguishes a short-lived amber
"Waking up the server" state from a genuine failure. Render also uses an ephemeral local
filesystem for Free web services, so the SQLite audit file is suitable for a demo but is
not durable storage across restarts, deploys, or spin-downs. A production deployment would
move the audit store to durable managed storage and use a paid or otherwise persistent
service plan.

**Frontend: Vercel Free plan.** The static HTML, CSS, and JavaScript frontend is deployed as
a Vercel static site. Vercel provides the public frontend URL and serves the files globally;
it does not run the scoring model. The frontend calls the Render API through the configurable
`THEMIS_API_BASE` value in `frontend/config.template.js`; the injection script can generate
the deployment config from the Vercel environment. The legacy inline demo pages also retain
the same Render URL as a fallback, so replacing those fallbacks is a separate frontend
cleanup rather than part of this documentation-only task.

**Why this arrangement:** both services provide a practical no-card path for a zero-budget
hackathon demo, while keeping the decision engine and its audit chain in one backend. The
tradeoff is cold-start latency and ephemeral local state on Render's Free service, plus the
need to configure CORS and the API key for cross-origin Vercel requests. See `API.md` for
the deployed contract and integration examples.

**Explicitly rejected, and why:**
- *Supabase (hosted Postgres) free tier* — unnecessary for this SQLite-backed demo because
  the hash chain is the differentiator, not relational features.
- *Paid trials or card-gated infrastructure* — outside the project's no-card constraint.
- *A managed SMS/OTP provider for `/release/request-otp`* — the demo returns an OTP directly
  as `otp_demo_only`; a production provider would require separate security and compliance
  work.

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
| Render Free web service | No-card FastAPI hosting for LightGBM + SHAP, with cold starts and ephemeral local state |
| Vercel Free static site | Global hosting for the frontend, kept separate from the decision engine |
