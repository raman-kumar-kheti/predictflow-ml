"""Integration tests for the FastAPI churn prediction service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    with TestClient(app) as test_client:
        res = test_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_model_info_endpoint() -> None:
    res = client.get("/model/info")
    assert res.status_code == 200
    body = res.json()
    assert "best_model" in body
    assert "best_metrics" in body


def test_single_and_batch_prediction_endpoints(sample_customer_record: dict) -> None:
    res = client.post("/predict", json=sample_customer_record)
    assert res.status_code == 200
    pred = res.json()
    assert pred["customer_id"] == "TEST-0001"
    assert pred["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}

    batch_res = client.post(
        "/predict/batch",
        json={"customers": [sample_customer_record, {**sample_customer_record, "customerID": "TEST-0002"}]},
    )
    assert batch_res.status_code == 200
    batch_body = batch_res.json()
    assert batch_body["total_customers"] == 2
    assert len(batch_body["predictions"]) == 2


def test_drift_endpoint(sample_customer_record: dict) -> None:
    res = client.post("/drift/check", json={"customers": [sample_customer_record] * 5})
    assert res.status_code == 200
    body = res.json()
    assert "overall_drift_detected" in body
