"""
API tests. Run from the project root:

    pip install flask pytest
    pytest backend/tests -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import app  # noqa: E402


@pytest.fixture()
def client():
    return app.test_client()


def preset(client, preset_id):
    schema = client.get("/api/schema").get_json()
    return next(p for p in schema["presets"] if p["id"] == preset_id)["values"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/api/stats", "/api/metrics", "/api/features", "/api/roc", "/api/feature-cols", "/api/schema"])
def test_read_endpoints_return_json(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.get_json()


def test_stats_have_fields_used_by_frontend(client):
    s = client.get("/api/stats").get_json()
    for key in ("total_records", "high_value_customers", "churn_rate", "feature_count"):
        assert key in s


def test_no_month9_or_sep_features(client):
    cols = client.get("/api/feature-cols").get_json()
    assert not [c for c in cols if c.endswith("_9") or c.startswith("sep_")]


def test_feature_importances_sorted_and_labelled(client):
    items = client.get("/api/features").get_json()
    assert len(items) == 15
    values = [i["importance"] for i in items]
    assert values == sorted(values, reverse=True)
    assert all(i["label"] for i in items)


def test_predict_empty_body_uses_typical_customer(client):
    r = client.post("/api/predict", json={})
    body = r.get_json()
    assert r.status_code == 200
    assert 0 <= body["score"] <= 1
    assert body["score"] == body["baseline_score"]
    assert body["provided_count"] == 0
    assert body["factors"] == []


def test_predict_response_shape(client):
    body = client.post("/api/predict", json=preset(client, "churner")).get_json()
    for key in ("score", "prediction", "risk_level", "baseline_score", "threshold", "factors"):
        assert key in body
    assert body["prediction"] in ("Churn", "No Churn")
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_churner_example_scores_higher_than_loyal_example(client):
    churner = client.post("/api/predict", json=preset(client, "churner")).get_json()
    loyal = client.post("/api/predict", json=preset(client, "loyal")).get_json()
    assert churner["score"] > loyal["score"]
    assert churner["score"] > churner["baseline_score"]


def test_more_incoming_calls_lowers_risk(client):
    low = client.post("/api/predict", json={"total_ic_mou_8": 0}).get_json()["score"]
    high = client.post("/api/predict", json={"total_ic_mou_8": 800}).get_json()["score"]
    assert low > high


def test_unknown_fields_are_ignored_and_reported(client):
    body = client.post("/api/predict", json={"not_a_feature": 5}).get_json()
    assert body["ignored"] == ["not_a_feature"]


@pytest.mark.parametrize("bad", ["abc", None, True, "", [1], 1e12])
def test_bad_numbers_are_rejected(client, bad):
    r = client.post("/api/predict", json={"arpu_8": bad})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_non_object_body_is_rejected(client):
    assert client.post("/api/predict", json=[1, 2]).status_code == 400


def test_missing_body_is_rejected(client):
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400


def test_huge_body_is_rejected(client):
    r = client.post("/api/predict", data="x" * 100_000, content_type="application/json")
    assert r.status_code in (400, 413)


def test_unknown_route_returns_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.get_json()["error"]


def test_retrain_endpoint_is_gone(client):
    assert client.post("/api/retrain").status_code == 404
