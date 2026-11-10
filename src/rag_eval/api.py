"""API endpoints isolated per authenticated tenant.

The in-memory indexes are not durable and are not shared between workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock, RLock

from fastapi import Depends, FastAPI, HTTPException

from .engine import RAGEngine
from .models import (
    DocumentBatch,
    EvaluationRequest,
    EvaluationSummary,
    QueryRequest,
    QueryResponse,
)
from .security import require_tenant


@dataclass
class TenantState:
    engine: RAGEngine = field(default_factory=RAGEngine)
    lock: RLock = field(default_factory=RLock)


class TenantStore:
    """Per-tenant indexes with synchronized lazy creation and operations."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._tenants: dict[str, TenantState] = {}

    def get(self, tenant: str) -> TenantState:
        with self._lock:
            if tenant not in self._tenants:
                self._tenants[tenant] = TenantState()
            return self._tenants[tenant]


app = FastAPI(
    title="Production RAG Evaluation Platform",
    version="0.2.0",
    description="Authenticated, tenant-isolated in-memory RAG baseline.",
)
app.state.store = TenantStore()


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
    state = app.state.store.get(tenant)
    with state.lock:
        try:
            chunks = state.engine.ingest(batch.documents)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"documents": state.engine.document_count, "chunks": chunks}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest, tenant: str = Depends(require_tenant)
) -> QueryResponse:
    state = app.state.store.get(tenant)
    with state.lock:
        if state.engine.document_count == 0:
            raise HTTPException(
                status_code=409,
                detail="No documents are indexed. Add documents first.",
            )
        return state.engine.query(request.question, top_k=request.top_k)


@app.post("/evaluate", response_model=EvaluationSummary)
def evaluate(
    request: EvaluationRequest, tenant: str = Depends(require_tenant)
) -> EvaluationSummary:
    state = app.state.store.get(tenant)
    with state.lock:
        if state.engine.document_count == 0:
            raise HTTPException(
                status_code=409,
                detail="No documents are indexed. Add documents first.",
            )
        return state.engine.evaluate(request.cases)
