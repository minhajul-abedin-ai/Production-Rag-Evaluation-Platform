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


def test_duplicate_retrieved_ids_count_once_for_recall_and_precision():
    recall, precision, rr = retrieval_metrics(
        ["a", "a", "a", "b"],
        ["a", "b", "c"],
    )
    assert recall == 2 / 3
    assert precision == 2 / 4
    assert rr == 1.0


def test_duplicate_retrieved_ids_cannot_make_recall_exceed_one():
    recall, precision, rr = retrieval_metrics(["a", "a", "a"], ["a"])
    assert recall == 1.0
    assert precision == 1 / 3
    assert rr == 1.0


def test_duplicate_nonrelevant_ids_do_not_change_reciprocal_rank():
    recall, precision, rr = retrieval_metrics(["x", "x", "a"], ["a"])
    assert recall == 1.0
    assert precision == 1 / 3
    assert rr == 1 / 3
