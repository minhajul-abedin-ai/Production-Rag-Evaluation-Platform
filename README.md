# Production RAG Evaluation Platform

A production-minded Retrieval-Augmented Generation (RAG) engineering project focused on **retrieval quality, grounding, guardrails, evaluation and API delivery**.

> Recommended repository name: `production-rag-evaluation-platform`

This project is designed as a portfolio piece for **AI / Machine Learning Engineer** roles. It demonstrates the engineering around RAG systems rather than hiding everything behind a single LLM API call.

## Why this project

RAG systems can return fluent answers while still retrieving the wrong evidence, missing relevant documents or responding to malicious instructions embedded in user input. This project makes those failure modes measurable.

The platform provides:

- document ingestion and chunking;
- TF-IDF retrieval as a transparent zero-cost baseline;
- extractive grounded answer generation;
- prompt-injection screening;
- citation-style source tracking;
- retrieval evaluation with Recall@K, Precision@K and MRR;
- groundedness and context-coverage diagnostics;
- FastAPI endpoints;
- automated tests and CI;
- Docker deployment.

The architecture is deliberately modular so a dense embedding model, vector database or external LLM can be added later without rewriting the evaluation layer.

## Architecture

```text
Documents
   |
   v
Chunker
   |
   v
Retriever (TF-IDF baseline)
   |
   +-------------------+
   |                   |
   v                   v
Guardrails        Evaluation Engine
   |                   |
   v                   +--> Recall@K
Grounded Answer        +--> Precision@K
   |                   +--> MRR
   v                   +--> Groundedness
FastAPI                +--> Context coverage
```

## Core engineering features

### Retrieval
The baseline retriever uses `TfidfVectorizer` and cosine similarity. A transparent lexical retriever is useful here because its behaviour is easy to inspect and benchmark.

### Grounded answer generation
The default answerer is extractive. It selects high-overlap sentences from retrieved context instead of inventing unsupported information. This keeps the repository fully runnable without paid model credentials.

### Guardrails
Queries are screened for common prompt-injection patterns such as attempts to override instructions, reveal system prompts or ignore policies.

### Evaluation
The evaluation module measures:

- **Recall@K** — how many relevant documents were retrieved;
- **Precision@K** — how much retrieved context was relevant;
- **MRR** — how early the first relevant result appeared;
- **Context coverage** — overlap between the question and retrieved evidence;
- **Groundedness** — lexical support for answer terms in retrieved evidence.

## Project structure

```text
.
├── .github/workflows/ci.yml
├── data/
│   ├── benchmark.json
│   └── sample_documents.json
├── src/
│   └── rag_eval/
│       ├── __init__.py
│       ├── answerer.py
│       ├── api.py
│       ├── chunking.py
│       ├── engine.py
│       ├── evaluation.py
│       ├── guardrails.py
│       ├── models.py
│       └── retriever.py
├── tests/
│   ├── test_api.py
│   ├── test_engine.py
│   └── test_evaluation.py
├── .dockerignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn rag_eval.api:app --app-dir src --reload
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

## Example API flow

### Add documents

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"doc_id":"rag-1","text":"Retrieval-Augmented Generation combines retrieval with generation.","metadata":{"topic":"rag"}}]}'
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?","top_k":3}'
```

### Run a benchmark

```bash
python -m rag_eval.engine --documents data/sample_documents.json --benchmark data/benchmark.json
```

## Tests

```bash
pytest -q
```

CI validates:

- unit/API tests;
- Python compilation;
- Docker image build.

## Roadmap

The current implementation intentionally starts with a transparent, free baseline. Natural extensions include:

- sentence-transformer embeddings;
- FAISS, pgvector or Qdrant;
- hybrid sparse+dense retrieval;
- reranking;
- LLM-as-a-judge evaluation;
- OpenAI/Anthropic/local-model adapters;
- tracing and observability;
- offline experiment comparison.

## Portfolio positioning

This project complements my ML/XAI work by demonstrating **production-minded GenAI/RAG engineering**: retrieval, evaluation, guardrails, APIs, testing, containerisation and reproducibility.

## Author

**Minhajul Abedin**  
AI / Machine Learning Engineer · London, United Kingdom

[GitHub](https://github.com/minhajul-abedin-ai) · [LinkedIn](https://www.linkedin.com/in/minhajulabedinai)
