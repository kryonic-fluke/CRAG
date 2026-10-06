import pytest
from fastapi.testclient import TestClient

from app.main import app
from eval.eval_suite import BENCHMARK_DATASET


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "llm_provider" in data


def test_query_endpoint_success(client):
    payload = {"question": "What is the equipment stipend policy?"}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "retry_count" in data


def test_query_endpoint_pydantic_validation_error(client):
    payload = {"question": "a"}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 422


def test_streaming_query_endpoint(client):
    payload = {"question": "What is Acme Corp's PTO policy?"}
    response = client.post("/api/v1/query/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert "data:" in body
    assert "done" in body


def test_eval_suite_dataset_integrity():
    assert len(BENCHMARK_DATASET) == 15
    categories = [item["category"] for item in BENCHMARK_DATASET]
    assert categories.count("In-Domain Policy") == 5
    assert categories.count("In-Domain Architecture") == 5
    assert categories.count("Out-of-Domain External") == 5

