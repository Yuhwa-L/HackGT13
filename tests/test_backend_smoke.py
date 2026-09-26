from fastapi.testclient import TestClient

from backend.main import app


def test_health():
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "any_mock" in body


def test_predict_is_503_without_model():
    assert TestClient(app).post("/api/predict").status_code == 503
