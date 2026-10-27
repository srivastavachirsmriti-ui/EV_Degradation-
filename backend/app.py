"""Flask REST API for the EV Battery Degradation Complaint Mining dashboard.

Endpoints
---------
GET  /api/health      service + model + dataset status
POST /api/predict     classify + cluster a new complaint
GET  /api/analytics   dashboard statistics derived from the dataset
GET  /api/complaints  searchable/filterable complaint table (paginated)
GET  /api/filters     distinct filter values for the frontend

The API only *reads* the existing models and CSV; it never retrains or
overwrites them.
"""

from __future__ import annotations

import logging
import math

import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS

import analytics
import config
from model_service import model_service
from text_cleaning import clean_text


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ev-api")

app = Flask(__name__)
app.json.sort_keys = False
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024
CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

MAX_TEXT_LENGTH = 5000


def error(message: str, status: int, details: str | None = None):
    body = {"error": True, "message": message, "status": status}
    if details:
        body["details"] = details
    return jsonify(body), status


@app.get("/api/health")
def health():
    info = {
        "status": "ok",
        "sklearn_version": sklearn.__version__,
        "expected_sklearn_version": config.EXPECTED_SKLEARN_VERSION,
        "sklearn_version_match": sklearn.__version__ == config.EXPECTED_SKLEARN_VERSION,
        "models": {},
        "dataset": {},
    }

    try:
        bundle = model_service.load()
        info["models"] = {
            "loaded": True,
            "classifier_classes": bundle.classes,
            "n_clusters": bundle.n_clusters,
            "supports_probabilities": bundle.supports_probabilities,
            "warnings": bundle.warnings,
        }
        regressor = model_service.load_regressor()
        info["models"]["regressor_loaded"] = True
        info["models"]["regressor"] = regressor["selected_model"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Model load failed")
        info["status"] = "degraded"
        info["models"] = {"loaded": False, "error": str(exc)}

    try:
        df = analytics.load_dataframe()
        info["dataset"] = {"loaded": True, "rows": int(len(df)), "path": str(config.DATA_PATH)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Dataset load failed")
        info["status"] = "degraded"
        info["dataset"] = {"loaded": False, "error": str(exc)}

    status_code = 200 if info["status"] == "ok" else 503
    return jsonify(info), status_code


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error("Request body must be a JSON object with a 'text' field.", 400)

    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return error("Field 'text' is required and must be a non-empty string.", 400)
    if len(text) > MAX_TEXT_LENGTH:
        return error(f"Field 'text' exceeds the {MAX_TEXT_LENGTH} character limit.", 400)

    include_cluster = payload.get("include_cluster", True)
    if not isinstance(include_cluster, bool):
        return error("Field 'include_cluster' must be a boolean.", 400)
    if not any(c.isalnum() for c in clean_text(text)):
        return error("Field 'text' must contain readable letters or numbers.", 400)
    try:
        result = model_service.predict(text, include_cluster=include_cluster)
    except FileNotFoundError as exc:
        return error("A required model file is missing.", 503)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        return error("Prediction failed.", 500)

    labels = analytics.cluster_labels()
    meta = labels.get(result["cluster_id"], {})
    result["cluster_label"] = (
        f"Cluster {result['cluster_id']} \u2013 {meta['dominant_issue']}"
        if meta.get("dominant_issue")
        else f"Cluster {result['cluster_id']}" if result["cluster_id"] is not None else "Clustering skipped"
    )
    result["cluster_dominant_issue"] = meta.get("dominant_issue")
    result["cluster_dominant_share"] = meta.get("dominant_share")
    return jsonify(result), 200


@app.post("/api/predict-batch")
def predict_batch():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error("Request body must be a JSON object.", 400)
    texts = payload.get("texts")
    include_cluster = payload.get("include_cluster", False)
    if not isinstance(texts, list) or not 1 <= len(texts) <= 100:
        return error("Field 'texts' must be a list of 1 to 100 complaints.", 400)
    if not isinstance(include_cluster, bool):
        return error("Field 'include_cluster' must be a boolean.", 400)
    if any(not isinstance(t, str) or not t.strip() or len(t) > MAX_TEXT_LENGTH
           or not any(c.isalnum() for c in clean_text(t)) for t in texts):
        return error("Each complaint must contain readable text and at most 5000 characters.", 400)
    try:
        return jsonify({"predictions": model_service.predict_many(texts, include_cluster), "count": len(texts)}), 200
    except Exception:
        logger.exception("Batch prediction failed")
        return error("Batch prediction failed.", 503)


REGRESSION_LIMITS = {
    "vehicle_age_months": (0, 360), "total_charging_cycles": (0, 10000),
    "avg_temperature_c": (-50, 85), "fast_charge_ratio": (0, 1),
    "avg_discharge_rate_c": (0, 10),
}


@app.post("/api/predict-degradation")
def predict_degradation():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error("Request body must be a JSON object with five numerical inputs.", 400)
    if set(payload) != set(REGRESSION_LIMITS):
        return error("Provide exactly these inputs: " + ", ".join(REGRESSION_LIMITS), 400)
    for key, (low, high) in REGRESSION_LIMITS.items():
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            return error(f"'{key}' must be a finite number between {low} and {high}.", 400)
    try:
        return jsonify(model_service.predict_degradation(payload)), 200
    except Exception:
        logger.exception("Degradation prediction failed")
        return error("Degradation model unavailable.", 503)


@app.get("/api/analytics")
def analytics_endpoint():
    try:
        return jsonify(analytics.analytics_payload()), 200
    except FileNotFoundError as exc:
        return error("Dataset file is missing.", 500, str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Analytics failed")
        return error("Failed to compute analytics.", 500, str(exc))


@app.get("/api/complaints")
def complaints_endpoint():
    try:
        limit = int(request.args.get("limit", 25))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return error("'limit' and 'offset' must be integers.", 400)

    cluster_raw = request.args.get("cluster_id")
    cluster_id = None
    if cluster_raw not in (None, "", "all"):
        try:
            cluster_id = int(cluster_raw)
        except ValueError:
            return error("'cluster_id' must be an integer or 'all'.", 400)

    try:
        payload = analytics.complaints_payload(
            search=request.args.get("search"),
            category=request.args.get("category"),
            severity=request.args.get("severity"),
            battery_type=request.args.get("battery_type"),
            cluster_id=cluster_id,
            limit=limit,
            offset=offset,
        )
        return jsonify(payload), 200
    except FileNotFoundError as exc:
        return error("Dataset file is missing.", 500, str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Complaint query failed")
        return error("Failed to query complaints.", 500, str(exc))


@app.get("/api/filters")
def filters_endpoint():
    try:
        return jsonify(analytics.filter_options()), 200
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Filter options failed")
        return error("Failed to load filter options.", 500, str(exc))


@app.errorhandler(404)
def not_found(_):
    return error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return error("Method not allowed for this endpoint.", 405)


@app.errorhandler(413)
def too_large(_):
    return error("Request body exceeds the 512 KiB limit.", 413)


if __name__ == "__main__":
    try:
        model_service.load()
        logger.info("Models loaded successfully.")
    except Exception as exc:
        logger.error("Could not preload models: %s", exc)
    logger.info("Starting API on http://%s:%s", config.HOST, config.PORT)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
