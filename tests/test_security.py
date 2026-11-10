"""Security regression tests: no unauthenticated or cross-tenant access."""
import json

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import TenantStore, app

ALICE_KEY = "alice-" + "a" * 40
BOB_KEY = "bob-" + "b" * 40


@pytest.fixture(autouse=True)
def configured_tenants(monkeypatch):
    monkeypatch.setenv(
        "RAG_API_KEYS_JSON", json.dumps({"alice": ALICE_KEY, "bob": BOB_KEY})
    )
    app.state.store = TenantStore()


def client() -> TestClient:
    return TestClient(app)


def auth(key: str = ALICE_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def ingest(c: TestClient, doc_id="rag", text="RAG combines retrieval with generation.", key=ALICE_KEY):
    return c.post(
        "/documents", headers=auth(key),
        json={"documents": [{"doc_id": doc_id, "text": text, "metadata": {"topic": "rag"}}]},
    )


def test_health_requires_bearer_and_never_exposes_counts_anonymously():
    c = client()
    response = c.get("/health")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert c.get("/health", headers=auth()).json() == {
        "status": "ok", "documents": 0, "chunks": 0,
    }


@pytest.mark.parametrize("path,payload", [
    ("/documents", {"documents": [{"doc_id": "rag", "text": "Private research document."}]}),
    ("/query", {"question": "What is RAG?"}),
    ("/evaluate", {"cases": [{"question": "What is RAG?", "relevant_doc_ids": ["rag"]}]}),
])
def test_data_endpoints_reject_missing_and_invalid_tokens(path, payload):
    c = client()
    assert c.post(path, json=payload).status_code == 401
    assert c.post(path, json=payload, headers=auth("wrong-token")).status_code == 401


def test_query_requires_own_documents():
    c = client()
    assert c.post("/query", headers=auth(), json={"question": "What is RAG?"}).status_code == 409


def test_isolation_with_identical_document_ids_and_cross_tenant_evaluation():
    c = client()
    assert ingest(c, "shared", "Alice private astronomy research", ALICE_KEY).status_code == 200
    assert c.get("/health", headers=auth(BOB_KEY)).json()["documents"] == 0
    assert c.post("/query", headers=auth(BOB_KEY), json={"question": "astronomy research"}).status_code == 409

    assert ingest(c, "shared", "Bob confidential nutrition data", BOB_KEY).status_code == 200
    query_alice = c.post("/query", headers=auth(ALICE_KEY), json={"question": "astronomy research"})
    query_bob = c.post("/query", headers=auth(BOB_KEY), json={"question": "nutrition data"})
    assert query_alice.status_code == query_bob.status_code == 200
    assert "Alice" in query_alice.json()["sources"][0]["text"]
    assert "Bob" in query_bob.json()["sources"][0]["text"]
    assert "Bob" not in str(query_alice.json())
    assert "Alice" not in str(query_bob.json())

    response = c.post("/evaluate", headers=auth(ALICE_KEY), json={"cases": [{
        "question": "astronomy research", "relevant_doc_ids": ["shared"]
    }]})
    assert response.status_code == 200
    assert response.json()["mean_recall_at_k"] == 1.0


def test_invalid_ingest_does_not_corrupt_tenant_or_other_tenants():
    c = client()
    assert ingest(c, text="Real searchable reference material").status_code == 200
    assert ingest(c, "invalid", "the and or", BOB_KEY).status_code == 422
    assert c.get("/health", headers=auth()).json()["documents"] == 1
    assert c.get("/health", headers=auth(BOB_KEY)).json()["documents"] == 0


def test_authentication_fails_closed_without_keys(monkeypatch):
    c = client()
    monkeypatch.delenv("RAG_API_KEYS_JSON")
    assert c.get("/health").status_code == 503
    assert c.post("/documents", headers=auth(), json={"documents": [
        {"doc_id": "a", "text": "Test"}
    ]}).status_code == 503


@pytest.mark.parametrize("invalid", [
    "not-json", "{}", '{"alice": "short"}',
    json.dumps({"alice": ALICE_KEY, "bob": ALICE_KEY}),
])
def test_invalid_auth_config_fails_closed(monkeypatch, invalid):
    monkeypatch.setenv("RAG_API_KEYS_JSON", invalid)
    assert client().get("/health", headers=auth()).status_code == 503


def test_missing_authorization_never_creates_tenant_state():
    c = client()
    assert c.post("/documents", json={"documents": [
        {"doc_id": "private", "text": "Sensitive information"}
    ]}).status_code == 401
    assert app.state.store._tenants == {}
