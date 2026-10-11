from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentBatch(BaseModel):
    documents: list[Document] = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    guardrail_triggered: bool = False
    guardrail_reasons: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    question: str
    relevant_doc_ids: list[str] = Field(min_length=1)
    expected_answer: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)


class EvaluationResult(BaseModel):
    question: str
    retrieved_doc_ids: list[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    context_coverage: float
    groundedness: float


class EvaluationSummary(BaseModel):
    cases: list[EvaluationResult]
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float
    mean_context_coverage: float
    mean_groundedness: float
