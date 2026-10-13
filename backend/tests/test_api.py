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
    # SGDClassifier(hinge) must not pretend to give probabilities.
    assert body["probabilities"] is None
    assert body["supports_probabilities"] is False
    assert body["decision_scores"] is not None


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
    assert body["evaluation"] is None
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
