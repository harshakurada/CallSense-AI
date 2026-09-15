from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unimplemented_routes_return_501():
    resp = client.post("/analyze")
    assert resp.status_code == 501
