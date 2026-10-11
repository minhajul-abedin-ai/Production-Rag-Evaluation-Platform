from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk
from .models import SourceChunk


_QUERY_EXPANSIONS = {
    "rag": "retrieval augmented generation",
    "xai": "explainable artificial intelligence",
    "mrr": "mean reciprocal rank",
}


def _expand_query(query: str) -> str:
    """Expand common AI acronyms while preserving the original query terms."""
    expanded = query
    for acronym, phrase in _QUERY_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(acronym)}\b", query, flags=re.IGNORECASE):
            expanded = f"{expanded} {phrase}"
    return expanded


@dataclass
class RetrievalIndex:
    chunks: list[Chunk]
    vectorizer: TfidfVectorizer
    matrix: object


class TfidfRetriever:
    """Transparent sparse-retrieval baseline for RAG experiments."""

    def __init__(self) -> None:
        self._index: RetrievalIndex | None = None

    @property
    def indexed_chunks(self) -> int:
        return len(self._index.chunks) if self._index else 0

    def fit(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._index = None
            return

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
        self._index = RetrievalIndex(
            chunks=list(chunks),
            vectorizer=vectorizer,
            matrix=matrix,
        )

    def search(self, query: str, top_k: int = 3) -> list[SourceChunk]:
        if not self._index or not query.strip():
            return []

        normalized_query = _expand_query(query)
        query_vector = self._index.vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vector, self._index.matrix).ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SourceChunk] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self._index.chunks[int(index)]
            results.append(
                SourceChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=round(score, 6),
                    metadata=chunk.metadata,
                )
            )
        return results
