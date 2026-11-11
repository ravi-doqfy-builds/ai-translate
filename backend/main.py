"""
Root application — mounts all feature services under their prefixes.

Routes (all under /api/v1 to match frontend config.js):
  /api/v1/translation/...      → translation service
  /api/v1/summary/...          → summary service
  /api/v1/interact/...         → interact service
  /api/v1/comparison/...       → comparison service
  /api/v1/extract/...          → extract service
  /api/v1/transliteration/...  → transliteration service
  /api/v1/redact/...           → redact service
"""

import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Translate — Unified API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "ai-translate",
        "version": "1.0.0",
        "services": [
            "/api/v1/translation",
            "/api/v1/summary",
            "/api/v1/interact",
            "/api/v1/comparison",
            "/api/v1/extract",
            "/api/v1/transliteration",
            "/api/v1/redact",
        ],
    }


# ---------------------------------------------------------------------------
# Helper: temporarily put a service directory first on sys.path so its
# local imports (llm_client, doc_reader, etc.) resolve correctly, then
# import its app/router and restore the path.
# ---------------------------------------------------------------------------
def _service_dir(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)


def _load_router(service_name: str):
    """Import the APIRouter from a service's router.py with its own sys.path."""
    svc_dir = _service_dir(service_name)
    sys.path.insert(0, svc_dir)
    try:
        import importlib
        mod = importlib.import_module(f"{service_name}.router")
        return mod.router
    finally:
        sys.path.remove(svc_dir)


def _load_app(service_name: str):
    """Import the FastAPI app from a service's main.py with its own sys.path."""
    svc_dir = _service_dir(service_name)
    sys.path.insert(0, svc_dir)
    try:
        import importlib
        mod = importlib.import_module(f"{service_name}.main")
        return mod.app
    finally:
        sys.path.remove(svc_dir)


# ---------------------------------------------------------------------------
# Mount services
# ---------------------------------------------------------------------------

# Translation has a monolithic main.py — mount its app directly.
try:
    translation_app = _load_app("translation")
    app.mount("/api/v1/translation", translation_app)
    print("[startup] mounted /api/v1/translation")
except Exception as exc:
    print(f"[startup] WARNING: translation failed to load: {exc}")

# All other services expose an APIRouter in router.py.
_ROUTER_SERVICES = [
    ("summary",         "/api/v1/summary"),
    ("interact",        "/api/v1/interact"),
    ("comparison",      "/api/v1/comparison"),
    ("extract",         "/api/v1/extract"),
    ("transliteration", "/api/v1/transliteration"),
    ("redact",          "/api/v1/redact"),
    ("dynamic_fields",  "/api/v1/dynamic-fields"),
]

for svc_name, prefix in _ROUTER_SERVICES:
    try:
        router = _load_router(svc_name)
        app.include_router(router, prefix=prefix)
        print(f"[startup] mounted {prefix}")
    except Exception as exc:
        print(f"[startup] WARNING: {svc_name} failed to load: {exc}")
