# Sentinel — HF Spaces (Docker SDK, CPU Basic, free tier)
FROM python:3.13-slim

# --- system deps ------------------------------------------------------------
# libgomp1: LightGBM's compiled wheel links against OpenMP at runtime; without
# this the import succeeds but training/inference calls can crash with a
# "libgomp.so.1: cannot open shared object file" error on slim Debian images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- python deps (leverage Docker layer caching) ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- project code ------------------------------------------------------------
COPY . .

# --- path fix: backend/serve.py does `from narration import get_narration`
# (a bare import, not `from backend.narration import ...`), which only
# resolves if backend/ itself is on sys.path. Adding it explicitly here
# means no source edit is needed to make the container import cleanly.
ENV PYTHONPATH=/app:/app/backend

# --- artifact locations: train.py writes these under ml/artifacts/, but
# serve.py's ModelBundle defaults to bare "model.joblib" / "metrics.json"
# in the CWD. Point the env vars it already reads at the real location so
# nothing needs to move or be duplicated.
ENV SENTINEL_MODEL_PATH=/app/ml/artifacts/sentinel_model.joblib
ENV SENTINEL_METRICS_PATH=/app/ml/artifacts/metrics.json
# feature_order.json is not produced anywhere in this repo today, so this
# is left unset deliberately — see the "known issues" note that ships
# alongside this Dockerfile. ModelBundle.load() will fall back to its
# hardcoded 8-column feature_order list when this file is absent.
# ENV SENTINEL_FEATURES_PATH=/app/ml/artifacts/feature_order.json

# Anthropic-unrelated demo API keys — set as HF Spaces "Repository secrets",
# never baked into the image. Listed here only as documentation of what the
# app expects to find in its environment at runtime.
# ENV GEMINI_API_KEY=   (set via HF Spaces Settings > Repository secrets)
# ENV GROQ_API_KEY=     (set via HF Spaces Settings > Repository secrets)

# HF Spaces' Docker SDK routes traffic to port 7860 specifically.
EXPOSE 7860

CMD ["uvicorn", "backend.serve:app", "--host", "0.0.0.0", "--port", "7860"]
