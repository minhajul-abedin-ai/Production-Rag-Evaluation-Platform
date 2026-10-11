# Evaluation Design

The project treats retrieval evaluation as a first-class engineering concern.

## Metrics

- **Recall@K** measures how many known relevant documents are recovered in the top K results.
- **Precision@K** measures how much of the retrieved set is relevant.
- **Reciprocal Rank** rewards systems that return the first relevant result earlier.
- **Context Coverage** estimates how much question terminology appears in retrieved evidence.
- **Groundedness** estimates how much answer terminology is supported by retrieved evidence.

## Why a lexical baseline?

The first implementation uses TF-IDF because it is:

- deterministic;
- fast;
- inspectable;
- free to run;
- useful as a baseline before adding embeddings.

The retriever also expands common AI acronyms such as RAG, XAI and MRR so that acronym-heavy user questions can match documents containing expanded terms.

## Benchmark data

`data/benchmark.json` defines questions and relevant document IDs.

Run:

```bash
python -m rag_eval.engine \
  --documents data/sample_documents.json \
  --benchmark data/benchmark.json
```

The output is JSON containing per-case metrics and aggregate means.

## Future evaluation extensions

A stronger research/production version could add:

- nDCG;
- MAP;
- semantic answer similarity;
- human relevance labels;
- LLM-as-a-judge evaluation;
- adversarial prompt-injection benchmarks;
- latency and cost tracking;
- comparison across sparse, dense and hybrid retrieval.
