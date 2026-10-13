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

import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS

import analytics
import config
from model_service import model_service


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ev-api")

app = Flask(__name__)
app.json.sort_keys = False
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
        }
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

    try:
        result = model_service.predict(text)
    except FileNotFoundError as exc:
        return error("A required model file is missing.", 500, str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        return error("Prediction failed.", 500, str(exc))

    labels = analytics.cluster_labels()
    meta = labels.get(result["cluster_id"], {})
    result["cluster_label"] = (
        f"Cluster {result['cluster_id']} \u2013 {meta['dominant_issue']}"
        if meta.get("dominant_issue")
        else f"Cluster {result['cluster_id']}"
    )
    result["cluster_dominant_issue"] = meta.get("dominant_issue")
    result["cluster_dominant_share"] = meta.get("dominant_share")
    return jsonify(result), 200


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


if __name__ == "__main__":
    try:
        model_service.load()
        logger.info("Models loaded successfully.")
    except Exception as exc:
        logger.error("Could not preload models: %s", exc)
    logger.info("Starting API on http://%s:%s", config.HOST, config.PORT)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
