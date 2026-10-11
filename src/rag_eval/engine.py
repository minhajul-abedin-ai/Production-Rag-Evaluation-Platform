from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answerer import build_grounded_answer
from .chunking import chunk_documents
from .evaluation import context_coverage, groundedness, retrieval_metrics, summarize
from .guardrails import inspect_query
from .models import (
    Document,
    EvaluationCase,
    EvaluationResult,
    EvaluationSummary,
    QueryResponse,
)
from .retriever import TfidfRetriever


class RAGEngine:
    def __init__(
        self,
        max_chunk_words: int = 120,
        overlap_words: int = 25,
    ) -> None:
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self._documents: dict[str, Document] = {}
        self.retriever = TfidfRetriever()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def ingest(self, documents: list[Document]) -> int:
        for document in documents:
            self._documents[document.doc_id] = document

        chunks = chunk_documents(
            list(self._documents.values()),
            max_words=self.max_chunk_words,
            overlap_words=self.overlap_words,
        )
        self.retriever.fit(chunks)
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> QueryResponse:
        guardrail = inspect_query(question)
        if guardrail.triggered:
            return QueryResponse(
                question=question,
                answer="Query blocked by the input guardrail.",
                sources=[],
                guardrail_triggered=True,
                guardrail_reasons=guardrail.reasons,
            )

        sources = self.retriever.search(question, top_k=top_k)
        answer = build_grounded_answer(question, sources)

        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
        )

    def evaluate(self, cases: list[EvaluationCase]) -> EvaluationSummary:
        results: list[EvaluationResult] = []

        for case in cases:
            response = self.query(case.question, top_k=case.top_k)
            retrieved_doc_ids: list[str] = []
            for source in response.sources:
                if source.doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(source.doc_id)

            recall, precision, reciprocal_rank = retrieval_metrics(
                retrieved_doc_ids,
                case.relevant_doc_ids,
            )
            contexts = [source.text for source in response.sources]

            results.append(
                EvaluationResult(
                    question=case.question,
                    retrieved_doc_ids=retrieved_doc_ids,
                    recall_at_k=round(recall, 6),
                    precision_at_k=round(precision, 6),
                    reciprocal_rank=round(reciprocal_rank, 6),
                    context_coverage=round(context_coverage(case.question, contexts), 6),
                    groundedness=round(groundedness(response.answer, contexts), 6),
                )
            )

        return summarize(results)


def _load_documents(path: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Document.model_validate(item) for item in payload]


def _load_cases(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(item) for item in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local RAG benchmark.")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()

    engine = RAGEngine()
    engine.ingest(_load_documents(args.documents))
    summary = engine.evaluate(_load_cases(args.benchmark))
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
