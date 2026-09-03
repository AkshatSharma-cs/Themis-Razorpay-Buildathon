# Themis API

## Live service

The deployed backend base URL is:

```text
https://themis-razorpay-buildathon.onrender.com
```

For the curl examples below:

```bash
export THEMIS_BASE_URL="https://themis-razorpay-buildathon.onrender.com"
```

Interactive OpenAPI documentation is available at `https://themis-razorpay-buildathon.onrender.com/docs`.
The deployable frontend configuration value is `THEMIS_API_BASE` in `frontend/config.template.js`,
and `frontend/scripts/inject-config.js` can generate `config.js` from the Vercel environment.
The legacy inline demo pages also retain the same Render URL as a fallback; this roadmap/documentation
task does not rewrite those page scripts.

Render's Free web service may sleep after 15 minutes without inbound traffic. The first request afterward can take about one minute while the service starts.

## Authentication and limits

Protected endpoints require this header:

```text
X-API-Key: themis-demo-key
```

The server accepts one or more comma-separated keys from `THEMIS_API_KEYS`. Missing or invalid keys receive HTTP `401` with:

```json
{"detail":{"error":"invalid_api_key","message":"Missing or invalid X-API-Key header."}}
```

Per-key in-memory rate limiting applies to `/v1/score`, `/v1/decision`, and `/v1/release/*`. The default is 60 requests per one-minute window, configurable with `THEMIS_RATE_LIMIT_PER_MINUTE`. Exceeding it returns HTTP `429`. The limiter resets when the process restarts and is not shared across multiple instances.

CORS is enabled for origins in the comma-separated `THEMIS_CORS_ORIGINS` environment variable. The current default is permissive (`*`) for this demo. `/health`, `/v1/health`, and audit read/verification endpoints are open; audit endpoints do not require an API key.

## Endpoint contract

All `/v1` routes below have unversioned compatibility aliases. FastAPI's `/docs` is the authoritative generated schema.

### `GET /health` and `GET /v1/health`

Open uptime/model status endpoint. Response shape:

```json
{"status":"ok","model_version":"loaded","is_dummy_model":false,"threshold":0.16,"n_features":28}
```

```bash
curl "$THEMIS_BASE_URL/health"
```

### `POST /v1/score`

Requires `X-API-Key`. Request fields include required `payer_vpa`, `payee_vpa`, and `amount`, plus optional transaction, device, payee, session, categorical, and `extra_features` fields. The full request schema is `TransactionPayload` in `/docs`. Response shape:

```json
{"tx_id":"txn_abc123","probability":0.23,"top_features":[{"feature":"amount","value":649992.0,"contribution":0.7}],"model_version":"loaded"}
```

```bash
curl -X POST "$THEMIS_BASE_URL/v1/score" -H "Content-Type: application/json" -H "X-API-Key: themis-demo-key" -d '{"payer_vpa":"payer@upi","payee_vpa":"merchant@upi","amount":850}'
```

### `POST /v1/decision`

Requires `X-API-Key` and uses the same `TransactionPayload` request as `/v1/score`. Response fields are `tx_id`, `probability`, `threshold`, `action_type`, `reason`, `duration_hours`, `verification`, `narration`, `narration_source`, and `audit_row_hash`.

```bash
curl -X POST "$THEMIS_BASE_URL/v1/decision" -H "Content-Type: application/json" -H "X-API-Key: themis-demo-key" -d '{"payer_vpa":"payer@upi","payee_vpa":"merchant@upi","amount":850}'
```

### `POST /v1/release/request-otp`

Requires `X-API-Key`. Request: `{"tx_id":"...","payer_vpa":"..."}`. Demo response: `{"tx_id":"...","otp_demo_only":"123456","expires_in_seconds":300}`.

```bash
curl -X POST "$THEMIS_BASE_URL/v1/release/request-otp" -H "Content-Type: application/json" -H "X-API-Key: themis-demo-key" -d '{"tx_id":"txn_abc123","payer_vpa":"payer@upi"}'
```

### `POST /v1/release/confirm`

Requires `X-API-Key`. Request: `{"tx_id":"...","payer_vpa":"...","otp_code":"...","confirm_intent":true}`. Response: `{"tx_id":"...","released":true,"reason":"..."}`.

```bash
curl -X POST "$THEMIS_BASE_URL/v1/release/confirm" -H "Content-Type: application/json" -H "X-API-Key: themis-demo-key" -d '{"tx_id":"txn_abc123","payer_vpa":"payer@upi","otp_code":"123456","confirm_intent":true}'
```

### `GET /v1/audit/{tx_id}`

Open read endpoint. Response shape: `{"tx_id":"txn_abc123","events":[...]}`. Each event includes its event type, timestamp, payload, previous hash, and row hash.

```bash
curl "$THEMIS_BASE_URL/v1/audit/txn_abc123"
```

The open verification endpoint is `GET /v1/audit/verify/chain`; it returns `ok`, `rows_checked`, and, when broken, the first broken row/reason.
