"""Durable tenant storage, bounded ingestion, key rotation and throttling tests."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import TenantStore, app
from rag_eval.limits import SlidingWindowLimiter

A = "a" * 40
B = "b" * 40


@pytest.fixture(autouse=True)
def configure(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS_JSON", json.dumps({"alice": A, "bob": B}))
    monkeypatch.delenv("RAG_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("RAG_STORAGE_PATH", raising=False)
    app.state.store = TenantStore()
    app.state.limiter = SlidingWindowLimiter()


def header(token=A):
    return {"Authorization": f"Bearer {token}"}


def add(c, doc_id, text, token=A):
    return c.post("/documents", headers=header(token), json={
        "documents": [{"doc_id": doc_id, "text": text}]
    })


def test_sqlite_survives_restart_and_isolates_tenants(tmp_path):
    path = tmp_path / "private" / "documents.sqlite3"
    app.state.store = TenantStore(str(path))
    c = TestClient(app)
    assert add(c, "shared", "Alice astronomy telescope facts").status_code == 200
    assert add(c, "shared", "Bob nutrition protein data", B).status_code == 200
    assert path.exists()

    app.state.store = TenantStore(str(path))  # Simulated application restart
    assert c.get("/health", headers=header()).json()["documents"] == 1
    assert c.get("/health", headers=header(B)).json()["documents"] == 1
    a = c.post("/query", headers=header(), json={"question": "astronomy telescope"})
    b = c.post("/query", headers=header(B), json={"question": "nutrition protein"})
    assert "Alice" in a.json()["sources"][0]["text"]
    assert "Bob" in b.json()["sources"][0]["text"]
    assert "Bob" not in str(a.json())
    assert "Alice" not in str(b.json())


def test_failed_index_update_does_not_change_disk(tmp_path):
    path = tmp_path / "data.sqlite3"
    app.state.store = TenantStore(str(path))
    c = TestClient(app)
    assert add(c, "main", "Durable astronomy telescope notes").status_code == 200
    assert add(c, "main", "the and or").status_code == 422
    app.state.store = TenantStore(str(path))
    assert c.get("/health", headers=header()).json()["documents"] == 1
    response = c.post("/query", headers=header(), json={"question": "astronomy telescope"})
    assert response.status_code == 200
    assert "Durable" in response.json()["sources"][0]["text"]


def test_sqlite_write_error_does_not_publish_in_memory(tmp_path, monkeypatch):
    app.state.store = TenantStore(str(tmp_path / "data.sqlite3"))
    c = TestClient(app)
    assert add(c, "first", "Existing astronomy reference").status_code == 200
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("simulated disk write failure")
    monkeypatch.setattr(app.state.store._storage, "replace", fail)
    with pytest.raises(sqlite3.OperationalError):
        add(c, "second", "New unpublished research text")
    state = app.state.store.get("alice")
    assert state.engine.document_count == 1


def test_count_quota_rejects_without_modifying_index(monkeypatch):
    monkeypatch.setattr("rag_eval.api.MAX_DOCUMENTS_PER_TENANT", 1)
    c = TestClient(app)
    assert add(c, "one", "First research record").status_code == 200
    assert add(c, "two", "Second research record").status_code == 413
    assert c.get("/health", headers=header()).json()["documents"] == 1


def test_byte_quota_rejects(monkeypatch):
    monkeypatch.setattr("rag_eval.api.MAX_DOCUMENT_BYTES_PER_TENANT", 100)
    c = TestClient(app)
    assert add(c, "large", "research " * 30).status_code == 413


def test_rate_limit_is_per_tenant_and_returns_retry_after():
    app.state.limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
    c = TestClient(app)
    assert c.get("/health", headers=header()).status_code == 200
    assert c.get("/health", headers=header()).status_code == 200
    blocked = c.get("/health", headers=header())
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert c.get("/health", headers=header(B)).status_code == 200


def test_rotating_key_file_rejects_old_key(tmp_path, monkeypatch):
    filename = tmp_path / "keys.json"
    filename.write_text(json.dumps({"alice": A}), encoding="utf-8")
    filename.chmod(0o600)
    monkeypatch.setenv("RAG_API_KEYS_FILE", str(filename))
    c = TestClient(app)
    assert c.get("/health", headers=header(A)).status_code == 200
    filename.write_text(json.dumps({"alice": B}), encoding="utf-8")
    filename.chmod(0o600)
    assert c.get("/health", headers=header(A)).status_code == 401
    assert c.get("/health", headers=header(B)).status_code == 200


def test_world_readable_secret_file_fails_closed(tmp_path, monkeypatch):
    import os
    filename = tmp_path / "keys.json"
    filename.write_text(json.dumps({"alice": A}), encoding="utf-8")
    filename.chmod(0o644)
    monkeypatch.setenv("RAG_API_KEYS_FILE", str(filename))
    if os.name == "posix":
        assert TestClient(app).get("/health", headers=header(A)).status_code == 503


def test_document_and_batch_size_validation():
    c = TestClient(app)
    too_long = "x" * 20_001
    assert add(c, "huge", too_long).status_code == 422
    docs = [{"doc_id": str(i), "text": "Sample text"} for i in range(21)]
    assert c.post("/documents", headers=header(), json={"documents": docs}).status_code == 422
