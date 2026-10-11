import re
from statistics import mean

from .models import EvaluationResult, EvaluationSummary


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2
    }


def retrieval_metrics(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> tuple[float, float, float]:
    relevant = set(relevant_doc_ids)
    if not relevant:
        raise ValueError("relevant_doc_ids cannot be empty")

    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant)
    recall = hits / len(relevant)
    precision = hits / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return recall, precision, reciprocal_rank


def context_coverage(question: str, contexts: list[str]) -> float:
    question_tokens = _content_tokens(question)
    if not question_tokens:
        return 0.0

    context_tokens = _content_tokens(" ".join(contexts))
    return len(question_tokens & context_tokens) / len(question_tokens)


def groundedness(answer: str, contexts: list[str]) -> float:
    answer_tokens = {
        token
        for token in _content_tokens(answer)
        if not token.startswith("doc")
    }
    if not answer_tokens:
        return 0.0

    context_tokens = _content_tokens(" ".join(contexts))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def summarize(results: list[EvaluationResult]) -> EvaluationSummary:
    if not results:
        raise ValueError("At least one evaluation result is required")

    return EvaluationSummary(
        cases=results,
        mean_recall_at_k=mean(item.recall_at_k for item in results),
        mean_precision_at_k=mean(item.precision_at_k for item in results),
        mean_reciprocal_rank=mean(item.reciprocal_rank for item in results),
        mean_context_coverage=mean(item.context_coverage for item in results),
        mean_groundedness=mean(item.groundedness for item in results),
    )
