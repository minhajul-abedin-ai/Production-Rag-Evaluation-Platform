"""Document erasure must be tenant-scoped and durable."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from rag_eval.api import TenantStore, app
from rag_eval.limits import SlidingWindowLimiter

ALICE = "a" * 40
BOB = "b" * 40


@pytest.fixture(autouse=True)
def configure(monkeypatch):
    monkeypatch.setenv("RAG_API_KEYS_JSON", json.dumps({"alice": ALICE, "bob": BOB}))
    monkeypatch.delenv("RAG_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("RAG_STORAGE_PATH", raising=False)
    app.state.limiter = SlidingWindowLimiter()


def auth(key):
    return {"Authorization": "Bearer " + key}


def add(c, key, doc_id, text):
    return c.post("/documents", headers=auth(key), json={"documents": [{
        "doc_id": doc_id, "text": text
    }]})


def test_delete_one_id_does_not_affect_other_tenant(tmp_path):
    path = str(tmp_path / "docs.db")
    app.state.store = TenantStore(path)
    c = TestClient(app)
    assert add(c, ALICE, "shared", "Alice astronomy research").status_code == 200
    assert add(c, BOB, "shared", "Bob nutrition research").status_code == 200
    assert c.delete("/documents/shared", headers=auth(ALICE)).status_code == 204
    assert c.delete("/documents/shared", headers=auth(ALICE)).status_code == 404
    assert c.get("/health", headers=auth(ALICE)).json()["documents"] == 0
    assert c.get("/health", headers=auth(BOB)).json()["documents"] == 1

    app.state.store = TenantStore(path)
    assert c.get("/health", headers=auth(ALICE)).json()["documents"] == 0
    result = c.post("/query", headers=auth(BOB),
                    json={"question": "nutrition research"})
    assert result.status_code == 200
    assert "Bob" in result.json()["sources"][0]["text"]


def test_delete_all_only_clears_current_tenant(tmp_path):
    app.state.store = TenantStore(str(tmp_path / "docs.db"))
    c = TestClient(app)
    assert add(c, ALICE, "one", "First astronomy document").status_code == 200
    assert add(c, ALICE, "two", "Second astronomy document").status_code == 200
    assert add(c, BOB, "one", "Other tenant astronomy document").status_code == 200
    assert c.delete("/documents", headers=auth(ALICE)).status_code == 204
    assert c.get("/health", headers=auth(ALICE)).json()["documents"] == 0
    assert c.get("/health", headers=auth(BOB)).json()["documents"] == 1


def test_delete_requires_valid_bearer(tmp_path):
    app.state.store = TenantStore(str(tmp_path / "docs.db"))
    c = TestClient(app)
    assert add(c, ALICE, "one", "Confidential research notes").status_code == 200
    assert c.delete("/documents/one").status_code == 401
    assert c.delete("/documents", headers=auth("bad")).status_code == 401
    assert c.get("/health", headers=auth(ALICE)).json()["documents"] == 1


def test_delete_disk_failure_preserves_existing_index(tmp_path, monkeypatch):
    app.state.store = TenantStore(str(tmp_path / "docs.db"))
    c = TestClient(app)
    assert add(c, ALICE, "one", "Private astronomy study").status_code == 200
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("simulated persistence error")
    monkeypatch.setattr(app.state.store._storage, "replace", fail)
    with pytest.raises(sqlite3.OperationalError):
        c.delete("/documents/one", headers=auth(ALICE))
    assert app.state.store.get("alice").engine.document_count == 1
