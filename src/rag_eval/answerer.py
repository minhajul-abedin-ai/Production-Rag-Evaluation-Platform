import re

from .models import SourceChunk


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2
    }


def build_grounded_answer(
    question: str,
    sources: list[SourceChunk],
    max_sentences: int = 3,
) -> str:
    if not sources:
        return "I could not find enough evidence in the indexed documents."

    question_terms = _tokens(question)
    candidates: list[tuple[float, str, str]] = []

    for source in sources:
        for sentence in _SENTENCE_RE.split(source.text):
            clean = sentence.strip()
            if not clean:
                continue
            sentence_terms = _tokens(clean)
            overlap = len(question_terms & sentence_terms)
            score = overlap + source.score
            candidates.append((score, clean, source.doc_id))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    seen: set[str] = set()
    for score, sentence, doc_id in candidates:
        if score <= 0 or sentence in seen:
            continue
        selected.append(f"{sentence} [{doc_id}]")
        seen.add(sentence)
        if len(selected) >= max_sentences:
            break

    if not selected:
        best = sources[0]
        return f"{best.text} [{best.doc_id}]"

    return " ".join(selected)
