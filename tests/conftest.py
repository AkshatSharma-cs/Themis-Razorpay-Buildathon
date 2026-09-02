"""
Shared pytest fixtures and sys.path setup for the Sentinel test suite.

Layout assumed:
    <repo_root>/backend/serve.py       (does `from narration import ...` --
                                         a BARE import, so backend/ itself
                                         must be on sys.path, not just repo
                                         root)
    <repo_root>/backend/audit_db.py
    <repo_root>/backend/narration.py
    <repo_root>/ml/generate.py         (does `from latent_process import
                                         ...` -- same story, ml/ must be on
                                         sys.path)
    <repo_root>/ml/latent_process.py
    <repo_root>/tests/conftest.py      (this file)

We add repo_root, repo_root/backend, and repo_root/ml to sys.path once,
here, so every test module can do e.g.:
    import backend.serve as serve
    import backend.audit_db as audit_db
    import generate
    import latent_process
without each test file repeating path surgery.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
ML_DIR = REPO_ROOT / "ml"

for _p in (REPO_ROOT, BACKEND_DIR, ML_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# Make sure no stray GEMINI_API_KEY / GROQ_API_KEY from the ambient
# environment causes backend/narration.py to attempt a real network call
# during tests -- we always want the deterministic template fallback here.
# This MUST happen before anything imports backend.serve / narration, which
# is why it lives at conftest module level (conftest.py is imported before
# any test module is collected).
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)
