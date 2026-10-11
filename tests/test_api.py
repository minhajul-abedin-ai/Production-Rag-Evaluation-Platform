from fastapi.testclient import TestClient

from rag_eval.api import app
from rag_eval.engine import RAGEngine


def client() -> TestClient:
    app.state.engine = RAGEngine()
    return TestClient(app)


def test_health():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_requires_documents():
    response = client().post(
        "/query",
        json={"question": "What is RAG?", "top_k": 3},
    )
    assert response.status_code == 409


def test_document_ingest_and_query():
    test_client = client()

    ingest = test_client.post(
        "/documents",
        json={
            "documents": [
                {
                    "doc_id": "rag",
                    "text": "RAG combines retrieval with generation.",
                    "metadata": {"topic": "rag"},
                }
            ]
        },
    )
    assert ingest.status_code == 200

    query = test_client.post(
        "/query",
        json={"question": "What does RAG combine?", "top_k": 3},
    )
    assert query.status_code == 200
    assert query.json()["sources"][0]["doc_id"] == "rag"
