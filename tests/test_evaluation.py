from rag_eval.evaluation import context_coverage, groundedness, retrieval_metrics


def test_retrieval_metrics():
    recall, precision, rr = retrieval_metrics(
        ["a", "b", "c"],
        ["b", "x"],
    )

    assert recall == 0.5
    assert precision == 1 / 3
    assert rr == 0.5


def test_context_coverage():
    score = context_coverage(
        "What is retrieval augmented generation",
        ["Retrieval augmented generation combines retrieval and generation."],
    )
    assert score > 0.5


def test_groundedness():
    score = groundedness(
        "Retrieval improves grounded answers.",
        ["Retrieval improves grounding by using external evidence."],
    )
    assert 0 <= score <= 1
