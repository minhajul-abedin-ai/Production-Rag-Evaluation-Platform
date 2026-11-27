"""Production RAG Evaluation Platform."""

__all__ = ["RAGEngine"]


def __getattr__(name: str):
    """Preserve ``from rag_eval import RAGEngine`` without eager module loading.

    Eagerly importing ``engine`` here preloads it before ``python -m
    rag_eval.engine`` executes, triggering a runpy RuntimeWarning.
    """
    if name == "RAGEngine":
        from .engine import RAGEngine

        return RAGEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
