"""API integration tests. Run from the backend/ directory: pytest -q"""

import pytest

from app import app
from text_cleaning import clean_text

CLASSES = {
    "Capacity Degradation",
    "Charging Issue",
    "Other",
    "Range/Performance",
    "Thermal Issue",
}


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_health_reports_models_and_dataset(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["models"]["loaded"] is True
    assert body["dataset"]["rows"] == 2500
    assert body["models"]["n_clusters"] == 5


def test_predict_returns_issue_and_cluster(client):
    response = client.post(
        "/api/predict",
        json={"text": "Fast charging on my Nissan Leaf is much slower than before."},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["issue_category"] in CLASSES
    assert isinstance(body["cluster_id"], int)
    assert 0 <= body["cluster_id"] < 5
    assert body["cleaned_text"] == clean_text(
        "Fast charging on my Nissan Leaf is much slower than before."
    )
    assert body["supports_probabilities"] is True
    assert set(body["probabilities"]) == CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1)
    assert body["decision_scores"] is None
    assert isinstance(body["needs_review"], bool)


def test_predict_rejects_empty_text(client):
    response = client.post("/api/predict", json={"text": "   "})
    assert response.status_code == 400
    assert response.get_json()["error"] is True


def test_predict_rejects_bad_body(client):
    response = client.post("/api/predict", json={"nope": 1})
    assert response.status_code == 400


def test_analytics_matches_dataset(client):
    response = client.get("/api/analytics")
    assert response.status_code == 200
    body = response.get_json()
    assert body["kpis"]["total_complaints"] == 2500
    assert body["dataset"]["origin"] == "synthetic"
    assert len(body["category_distribution"]) == 5
    assert len(body["cluster_distribution"]) == 5
    assert 0 <= body["evaluation"]["classification"]["accuracy"] <= 1
    assert len(body["age_trend"]) > 0


def test_complaints_search_and_pagination(client):
    response = client.get("/api/complaints?search=charging&limit=5")
    assert response.status_code == 200
    body = response.get_json()
    assert body["limit"] == 5
    assert body["count"] <= 5
    assert body["total"] >= body["count"]


def test_filters(client):
    response = client.get("/api/filters")
    assert response.status_code == 200
    body = response.get_json()
    assert set(body["categories"]) == CLASSES
    assert body["clusters"] == [0, 1, 2, 3, 4]


def test_unknown_endpoint(client):
    assert client.get("/api/does-not-exist").status_code == 404


def test_cleaning_is_reproducible():
    assert clean_text("Battery health is around 98.8%, and it's fine?") == (
        "battery health is around 98.8% and it s fine"
    )
    assert clean_text("The Mustang Mach-E uses VW ID.3 style cells; 3.5%.") == (
        "the mustang mach e uses vw id.3 style cells 3.5%."
    )


@pytest.mark.parametrize("text", ["...%%", "🔥🔥", 123, None, [], "x" * 5001])
def test_predict_rejects_invalid_text(client, text):
    assert client.post("/api/predict", json={"text": text}).status_code == 400


def test_batch_agrees_with_single_and_can_skip_clustering(client):
    texts = ["My battery becomes very hot during charging.", "Charging takes longer than usual."]
    batch = client.post("/api/predict-batch", json={"texts": texts}).get_json()
    assert batch["count"] == len(texts)
    for text, prediction in zip(texts, batch["predictions"]):
        single = client.post("/api/predict", json={"text": text, "include_cluster": False}).get_json()
        assert single["issue_category"] == prediction["issue_category"]
        assert single["probabilities"] == pytest.approx(prediction["probabilities"])
        assert single["cluster_id"] is None


@pytest.mark.parametrize("payload", [{}, {"texts": []}, {"texts": ["ok"] * 101},
    {"texts": ["charging", None]}, {"texts": "charging"}, {"texts": ["battery"], "include_cluster": "false"}])
def test_batch_validation(client, payload):
    assert client.post("/api/predict-batch", json=payload).status_code == 400


VALUES = {"vehicle_age_months": 48, "total_charging_cycles": 600,
          "avg_temperature_c": 28, "fast_charge_ratio": .3, "avg_discharge_rate_c": 1.2}


def test_degradation_estimate_and_interval(client):
    response = client.post("/api/predict-degradation", json=VALUES)
    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["interval_90"]["lower"] <= body["degradation_pct"] <= body["interval_90"]["upper"] <= 100
    assert "synthetic" in body["note"].lower()


@pytest.mark.parametrize("key,value", [("vehicle_age_months", -1), ("total_charging_cycles", True),
    ("fast_charge_ratio", 1.5), ("avg_temperature_c", "28"), ("avg_discharge_rate_c", None),
    ("avg_discharge_rate_c", float("inf")), ("avg_temperature_c", float("nan"))])
def test_degradation_rejects_invalid_numbers(client, key, value):
    assert client.post("/api/predict-degradation", json={**VALUES, key: value}).status_code == 400


def test_degradation_rejects_target_leakage_input(client):
    assert client.post("/api/predict-degradation", json={**VALUES, "battery_health_pct": 90}).status_code == 400


def test_degradation_flags_extrapolation(client):
    response = client.post("/api/predict-degradation", json={**VALUES, "vehicle_age_months": 200})
    assert response.status_code == 200
    assert "vehicle_age_months" in response.get_json()["out_of_training_range"]


def test_prediction_cache_cannot_be_mutated():
    from model_service import model_service
    first = model_service.predict("My battery is unusually hot.")
    first["probabilities"].clear()
    assert model_service.predict("My battery is unusually hot.")["probabilities"]


def test_unsupported_text_and_negation_require_review(client):
    for text in ["Pizza delivery today.", "No overheating, but charging is slow."]:
        result = client.post("/api/predict", json={"text": text}).get_json()
        assert result["needs_review"] is True


def test_template_and_regression_partitions_are_disjoint():
    import json
    from config import PROJECT_ROOT
    baseline = json.loads((PROJECT_ROOT / "benchmarks/baseline.json").read_text())
    assert not set(baseline["split"]["train_groups"]) & set(baseline["split"]["test_groups"])
    latest = json.loads((PROJECT_ROOT / "benchmarks/latest.json").read_text())
    split = latest["regression_split"]
    assert not set(split["fit_ids"]) & set(split["calibration_ids"])
    assert not set(split["test_ids"]) & (set(split["fit_ids"]) | set(split["calibration_ids"]))


def test_stale_metrics_are_not_displayed(monkeypatch, tmp_path):
    import analytics
    fake = tmp_path / "changed.csv"
    fake.write_text("changed")
    monkeypatch.setattr(analytics, "DATA_PATH", fake)
    assert analytics.evaluation_payload() is None


def test_age_buckets_cover_all_records():
    import analytics
    df = analytics.load_dataframe()
    assert sum(row["count"] for row in analytics._age_trend(df)) == len(df)
