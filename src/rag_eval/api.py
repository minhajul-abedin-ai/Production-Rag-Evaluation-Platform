from fastapi import FastAPI, HTTPException

from .engine import RAGEngine
from .models import (
    DocumentBatch,
    EvaluationRequest,
    EvaluationSummary,
    QueryRequest,
    QueryResponse,
)


app = FastAPI(
    title="Production RAG Evaluation Platform",
    version="0.1.0",
    description="Retrieval, guardrails and evaluation for a transparent RAG baseline.",
)
app.state.engine = RAGEngine()


@app.get("/health")
def health() -> dict[str, int | str]:
    engine: RAGEngine = app.state.engine
    return {
        "status": "ok",
        "documents": engine.document_count,
        "chunks": engine.retriever.indexed_chunks,
    }


@app.post("/documents")
def add_documents(batch: DocumentBatch) -> dict[str, int]:
    engine: RAGEngine = app.state.engine
    chunks = engine.ingest(batch.documents)
    return {
        "documents": engine.document_count,
        "chunks": chunks,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    engine: RAGEngine = app.state.engine
    if engine.document_count == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed. Add documents first.",
        )
    return engine.query(request.question, top_k=request.top_k)


@app.post("/evaluate", response_model=EvaluationSummary)
def evaluate(request: EvaluationRequest) -> EvaluationSummary:
    engine: RAGEngine = app.state.engine
    if engine.document_count == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents are indexed. Add documents first.",
        )
    return engine.evaluate(request.cases)
