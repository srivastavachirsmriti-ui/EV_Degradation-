"""Central configuration for the Flask backend.

Model/dataset paths point at the *existing* files in the project root by
default. Nothing is copied or moved; every path can be overridden with an
environment variable so the assets stay exactly where you keep them.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/config.py -> parents[1] == project root (ev_dashboard_bundle/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(var_name: str, default: Path) -> Path:
    raw = os.environ.get(var_name)
    return Path(raw).expanduser().resolve() if raw else default


# --- Input assets (existing trained models + dataset) -----------------------
DATA_PATH = _path_from_env("EV_DATA_PATH", PROJECT_ROOT / "dashboard_data.csv")
CLASSIFIER_PATH = _path_from_env("EV_CLASSIFIER_PATH", PROJECT_ROOT / "final_classifier.joblib")
CLUSTER_VECTORIZER_PATH = _path_from_env(
    "EV_CLUSTER_VECTORIZER_PATH", PROJECT_ROOT / "cluster_tfidf_vectorizer.joblib"
)
KMEANS_PATH = _path_from_env("EV_KMEANS_PATH", PROJECT_ROOT / "kmeans_model.joblib")
REGRESSOR_PATH = _path_from_env("EV_REGRESSOR_PATH", PROJECT_ROOT / "degradation_regressor.joblib")
EVALUATION_PATH = _path_from_env("EV_EVALUATION_PATH", PROJECT_ROOT / "benchmarks/latest.json")

# --- Server -----------------------------------------------------------------
HOST = os.environ.get("EV_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("EV_API_PORT", "5001"))
DEBUG = os.environ.get("EV_API_DEBUG", "0") == "1"

# Comma-separated list of allowed origins for the local Vite dev server.
_cors_default = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://127.0.0.1:4173"
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("EV_CORS_ORIGINS", _cors_default).split(",")
    if origin.strip()
]

# Expected scikit-learn major.minor used when the models were pickled.
EXPECTED_SKLEARN_VERSION = "1.6.1"
