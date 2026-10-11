from dataclasses import dataclass
from typing import Any

from .models import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


def chunk_document(
    document: Document,
    max_words: int = 120,
    overlap_words: int = 25,
) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be >= 0 and < max_words")

    words = document.text.split()
    if not words:
        return []

    step = max_words - overlap_words
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(words), step)):
        segment = words[start : start + max_words]
        if not segment:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::chunk-{index}",
                doc_id=document.doc_id,
                text=" ".join(segment),
                metadata=dict(document.metadata),
            )
        )
        if start + max_words >= len(words):
            break

    return chunks


def chunk_documents(
    documents: list[Document],
    max_words: int = 120,
    overlap_words: int = 25,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_words=max_words,
                overlap_words=overlap_words,
            )
        )
    return chunks
