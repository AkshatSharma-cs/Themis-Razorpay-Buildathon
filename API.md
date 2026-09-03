# Themis API

## Base URL

Use the deployed service URL as the base URL, for example:

```text
https://themis-razorpay-buildathon.onrender.com
```

Interactive OpenAPI documentation is available at `/docs`.

## Authentication

Protected endpoints require an `X-API-Key` header. Configure one or more comma-separated keys with `THEMIS_API_KEYS`.

```text
X-API-Key: themis-demo-key
```

`/health`, `/v1/health`, and audit read endpoints are open for monitoring and inspection. Requests to `/v1/score`, `/v1/decision`, and `/v1/release/*` are limited per API key by `THEMIS_RATE_LIMIT_PER_MINUTE` (default: 60).

CORS origins can be configured with the comma-separated `THEMIS_CORS_ORIGINS` environment variable. The default is permissive for initial integration.

## Endpoints

### Score

```bash
curl -X POST "$THEMIS_BASE_URL/v1/score" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: themis-demo-key" \
  -d '{"payer_vpa":"payer@upi","payee_vpa":"merchant@upi","amount":850}'
```

### Decision

```bash
curl -X POST "$THEMIS_BASE_URL/v1/decision" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: themis-demo-key" \
  -d '{"payer_vpa":"payer@upi","payee_vpa":"merchant@upi","amount":850}'
```

### Request release OTP

```bash
curl -X POST "$THEMIS_BASE_URL/v1/release/request-otp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: themis-demo-key" \
  -d '{"tx_id":"txn_abc123","payer_vpa":"payer@upi"}'
```

### Confirm release

```bash
curl -X POST "$THEMIS_BASE_URL/v1/release/confirm" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: themis-demo-key" \
  -d '{"tx_id":"txn_abc123","payer_vpa":"payer@upi","otp_code":"123456","confirm_intent":true}'
```

### Read an audit trail

```bash
curl "$THEMIS_BASE_URL/v1/audit/txn_abc123"
```

The original unversioned paths remain available as compatibility aliases.
