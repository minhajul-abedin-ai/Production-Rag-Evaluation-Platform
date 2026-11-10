"""Authenticated single-process RAG API with per-tenant indexes and optional persistence."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from threading import Lock, RLock

from fastapi import Depends, FastAPI, HTTPException

from .engine import RAGEngine
from .limits import (
    MAX_DOCUMENTS_PER_TENANT,
    MAX_DOCUMENT_BYTES_PER_TENANT,
    SlidingWindowLimiter,
)
from .models import (
    Document, DocumentBatch, EvaluationRequest, EvaluationSummary,
    QueryRequest, QueryResponse,
)
from .persistence import SQLiteDocumentStore
from .security import require_tenant


@dataclass
class TenantState:
    engine: RAGEngine = field(default_factory=RAGEngine)
    lock: RLock = field(default_factory=RLock)


class TenantStore:
    """Per-tenant engines; updates publish only after durable commit."""

    def __init__(self, storage_path: str | None = None) -> None:
        self._lock = Lock()
        self._tenants: dict[str, TenantState] = {}
        # Default remains ephemeral for unit tests and offline demos.
        if storage_path is None:
            storage_path = os.getenv("RAG_STORAGE_PATH", "")
        self._storage = SQLiteDocumentStore(storage_path) if storage_path else None

    def get(self, tenant: str) -> TenantState:
        with self._lock:
            if tenant not in self._tenants:
                engine = RAGEngine()
                if self._storage:
                    docs = self._storage.load(tenant)
                    if docs:
                        engine.ingest(docs)
                self._tenants[tenant] = TenantState(engine=engine)
            return self._tenants[tenant]

    def ingest(self, tenant: str, documents: list[Document]) -> dict[str, int]:
        state = self.get(tenant)
        with state.lock:
            candidate_docs = dict(state.engine._documents)
            candidate_docs.update((doc.doc_id, doc) for doc in documents)
            if len(candidate_docs) > MAX_DOCUMENTS_PER_TENANT:
                raise HTTPException(status_code=413, detail="Tenant document count limit exceeded")
            total_bytes = sum(len(d.model_dump_json().encode("utf-8"))
                              for d in candidate_docs.values())
            if total_bytes > MAX_DOCUMENT_BYTES_PER_TENANT:
                raise HTTPException(status_code=413, detail="Tenant document size limit exceeded")

            candidate_engine = RAGEngine(
                max_chunk_words=state.engine.max_chunk_words,
                overlap_words=state.engine.overlap_words,
            )
            try:
                chunks = candidate_engine.ingest(list(candidate_docs.values()))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            # If disk write fails, leave the existing in-memory index untouched.
            if self._storage:
                self._storage.replace(tenant, list(candidate_docs.values()))
            state.engine = candidate_engine
            return {"documents": candidate_engine.document_count, "chunks": chunks}


app = FastAPI(
    title="Production RAG Evaluation Platform",
    version="0.3.0",
    description="Authenticated, bounded single-process tenant RAG baseline.",
)
app.state.store = TenantStore()
app.state.limiter = SlidingWindowLimiter()


@app.get("/health")
def health(tenant: str = Depends(require_tenant)) -> dict[str, int | str]:
    state = app.state.store.get(tenant)
    with state.lock:
        return {
            "status": "ok",
            "documents": state.engine.document_count,
            "chunks": state.engine.retriever.indexed_chunks,
        }


@app.post("/documents")
def add_documents(
    batch: DocumentBatch, tenant: str = Depends(require_tenant)
) -> dict[str, int]:
    return app.state.store.ingest(tenant, batch.documents)


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, tenant: str = Depends(require_tenant)
) -> QueryResponse:
    state = app.state.store.get(tenant)
    with state.lock:
        if state.engine.document_count == 0:
            raise HTTPException(status_code=409, detail="No documents are indexed. Add documents first.")
        return state.engine.query(request.question, top_k=request.top_k)


@app.post("/evaluate", response_model=EvaluationSummary)
def evaluate(
    request: EvaluationRequest, tenant: str = Depends(require_tenant)
) -> EvaluationSummary:
    state = app.state.store.get(tenant)
    with state.lock:
        if state.engine.document_count == 0:
            raise HTTPException(status_code=409, detail="No documents are indexed. Add documents first.")
        return state.engine.evaluate(request.cases)
