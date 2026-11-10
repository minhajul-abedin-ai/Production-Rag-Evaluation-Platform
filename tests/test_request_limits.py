"""HTTP request-size boundary tests."""
import json
from fastapi.testclient import TestClient

from rag_eval.api import TenantStore, app

KEY = "a" * 40
HEADERS = {"Authorization": f"Bearer {KEY}"}


def test_oversized_json_rejected_before_ingest(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS_JSON", json.dumps({"demo": KEY}))
    app.state.store = TenantStore()
    c = TestClient(app)
    payload = {"documents": [{"doc_id": "doc", "text": "Safe text",
                             "metadata": {"body": "x" * (600 * 1024)}}]}
    response = c.post("/documents", headers=HEADERS, json=payload)
    assert response.status_code == 413
    assert c.get("/health", headers=HEADERS).json()["documents"] == 0


def test_normal_request_still_accepted(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS_JSON", json.dumps({"demo": KEY}))
    app.state.store = TenantStore()
    c = TestClient(app)
    response = c.post("/documents", headers=HEADERS,
                      json={"documents": [{"doc_id": "short", "text": "A real document"}]})
    assert response.status_code == 200
