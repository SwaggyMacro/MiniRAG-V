from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_info_endpoint():
    response = client.get("/v1/info")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["code"] == 200
    assert "data" in json_resp
    
    data = json_resp["data"]
    assert "app_name" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "healthy"

def test_explain_endpoint():
    payload = {"query": "What is decoupling?", "context": {}}
    response = client.post("/v1/explain", json=payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["code"] == 200
    
    data = json_resp["data"]
    assert data["original_query"] == "What is decoupling?"
    assert "explanation" in data
    assert isinstance(data["confidence"], float)
