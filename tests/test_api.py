"""API contract tests: authenticated, tenant-isolated requests."""
import json

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import TenantStore, app

DEMO_KEY = "demo-" + "x" * 40
HEADERS = {"Authorization": f"Bearer {DEMO_KEY}"}


@pytest.fixture(autouse=True)
def reset_api(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS_JSON", json.dumps({"demo": DEMO_KEY}))
    app.state.store = TenantStore()


def client() -> TestClient:
    return TestClient(app)


def test_health():
    response = client().get("/health", headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 0, "chunks": 0}


def test_query_requires_documents():
    response = client().post(
        "/query", headers=HEADERS,
        json={"question": "What is RAG?", "top_k": 3},
    )
    assert response.status_code == 409


def test_document_ingest_and_query():
    test_client = client()
    ingest = test_client.post(
        "/documents",
        headers=HEADERS,
        json={"documents": [{"doc_id": "rag",
            "text": "RAG combines retrieval with generation.",
            "metadata": {"topic": "rag"}}]},
    )
    assert ingest.status_code == 200
    query = test_client.post(
        "/query", headers=HEADERS,
        json={"question": "What does RAG combine?", "top_k": 3},
    )
    assert query.status_code == 200
    assert query.json()["sources"][0]["doc_id"] == "rag"


def test_stopwords_only_document_returns_422_not_500():
    test_client = client()
    response = test_client.post("/documents", headers=HEADERS, json={
        "documents": [{"doc_id": "words", "text": "the and or"}]
    })
    assert response.status_code == 422
    assert test_client.get("/health", headers=HEADERS).json()["documents"] == 0
