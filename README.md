# Production RAG Evaluation Platform

A production-minded Retrieval-Augmented Generation (RAG) engineering project focused on **retrieval quality, grounding, guardrails, evaluation and API delivery**.

This project is designed as a portfolio piece for **AI / Machine Learning Engineer** roles. It demonstrates the engineering around RAG systems rather than hiding everything behind a single LLM API call.

## Get started in minutes

**For developers:** [run the local FastAPI service](#quick-start), test requests using the [example API flow](#example-api-flow), and [reproduce sample evaluation metrics](#tests). You can run the baseline without paid LLM keys.

**For ML practitioners:** inspect how retrieval relevance, grounding proxies and baseline guardrails are evaluated. The initial retriever is **TF-IDF**, and the answerer is **extractive**, not a hosted generative LLM; embedding retrieval and LLM integrations are future work.

**Related work:** [ConsensusTree-AI multi-agent code review](https://github.com/minhajul-abedin-ai/ConsensusTree-AI) · [Author's AI engineering portfolio](https://github.com/minhajul-abedin-ai)

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
- FastAPI endpoints protected by bearer API keys; 
- separate per-tenant indexes with optional durable SQLite storage (no cross-tenant document retrieval);
- per-tenant document quotas and in-process rate limiting;
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
│       ├── security.py
│       ├── persistence.py
│       ├── limits.py
│       ├── request_limits.py
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
export RAG_API_KEYS_JSON='{"demo":"replace-with-at-least-32-random-characters-very-long"}'
uvicorn rag_eval.api:app --app-dir src --reload
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

## Authentication and tenant isolation

**Breaking API change:** all routes (including `/health`) now require a bearer API key. Missing or malformed `RAG_API_KEYS_JSON` fails closed with HTTP 503. Unauthorized requests receive 401 and do not access or create indexes.

Set an environment variable before launching the server, mapping tenant IDs to distinct, high-entropy API keys of **at least 32 characters**:

```bash
export RAG_API_KEYS_JSON='{"demo":"replace-with-at-least-32-random-characters-very-long"}'
```

Generate actual keys with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`; store them in a secret manager, never Git or shell history. For each authenticated request include `Authorization: Bearer <YOUR_TENANT_KEY>`. Assign different keys to different tenants: a shared key is a shared identity. Use HTTPS for remote access.

Each tenant receives a distinct in-memory document index. Even identical document IDs are isolated across tenants. Configuration and index creation are synchronized, and operations on the same tenant are serialized to avoid interleaved writes. No user-selected tenant header is trusted.

**Important limits:** This remains a single-process educational server, not a fully production-ready multi-tenant SaaS. Data are lost on restart and are not shared between workers. The static JSON key map requires manual rotation; there is no identity provider, quotas/rate limiting, persisted storage, auditable access control or enterprise authorization. Behind a production gateway, add resource limits, monitoring, persistent tenant-scoped storage and key rotation before serving external customers. The public docs UI is not protected against viewing endpoint schemas, but data endpoints require authentication.

## Durable storage, quotas, and credential rotation

The service now supports **optional tenant-scoped SQLite persistence**. By default, `RAG_STORAGE_PATH` is unset and data remain **ephemeral** for the free local demo. Set an absolute path to persist documents and rebuild each tenant's retrieval index on restart:

```bash
mkdir -p .rag-data .rag-secrets
chmod 700 .rag-data .rag-secrets
export RAG_STORAGE_PATH="$(pwd)/.rag-data/documents.sqlite3"
# Use your own unique, random 32+-character token for each tenant.
python -c 'import secrets; print(secrets.token_urlsafe(32))'
# Store the JSON mapping {"demo": "<generated token>"} in .rag-secrets/keys.json
chmod 600 .rag-secrets/keys.json
export RAG_API_KEYS_FILE="$(pwd)/.rag-secrets/keys.json"
uvicorn rag_eval.api:app --app-dir src --host 127.0.0.1 --port 8000
```

`RAG_API_KEYS_FILE` takes precedence over `RAG_API_KEYS_JSON`. Replacing the private JSON file atomically permits key rotation: prior keys stop authenticating on the next request; tenant IDs remain stable, so indexed content stays with its tenant. Use a real secret manager for distributed deployments. Do not commit JSON credential maps or SQLite databases. Keys are compared in constant time where practical, but this is not comprehensive identity management.

The demo enforces **120 authenticated requests per tenant per rolling minute** (HTTP 429 and `Retry-After` on excess), **200 stored documents per tenant**, **2 MB of serialized documents per tenant**, **20 documents per ingest request**, **512 KiB per HTTP request body**, **20,000 characters per document** and **50 evaluation cases per request**. Failed quota or validation checks do not replace the live index or committed rows. Disk errors do not publish an in-memory update.

**Deployment limits:** SQLite contains unencrypted user content, so use encrypted storage and strict file/volume permissions. SQLite persistence and the limiter are **single-process only**; do not run multiple workers sharing this file or rely on limiter state across replicas. This does not implement durable distributed rate limits, retries, an identity provider, per-IP login throttling, storage backups, GDPR retention/deletion workflows, slow-client timeouts, tamper-resistant audit logs or a security certification. Protect API traffic using TLS and a properly configured ingress proxy. Add integration, load, resilience and penetration testing before any public production use.

## Tenant document deletion

The authenticated API supports **`DELETE /documents/{doc_id}`** for one document and **`DELETE /documents`** to clear all documents belonging to the bearer token's tenant (HTTP 204). A missing document returns 404. A tenant cannot remove another tenant's documents, even if the document ID is the same. Both endpoints persist the updated tenant index when `RAG_STORAGE_PATH` is enabled, and preserve the live index if the storage write fails.

Deletion removes rows from the active SQLite database but **is not secure forensic erasure**: backups, snapshots, WAL/journal remnants and filesystem data recovery require separate retention controls. Never claim certified data deletion without a documented storage lifecycle and independent verification.

## Example API flow

The API requires a tenant bearer token on **every** endpoint, including document ingestion and queries. Start the server using the authentication setup above, then set `RAG_DEMO_TOKEN` to the same secret configured for the `demo` tenant. Do not commit or paste real tokens into examples or logs.

```bash
read -r -s -p 'Demo tenant API token: ' RAG_DEMO_TOKEN; echo
export RAG_DEMO_TOKEN
```

### Add documents

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Authorization: Bearer ${RAG_DEMO_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"doc_id":"rag-1","text":"Retrieval-Augmented Generation combines retrieval with generation.","metadata":{"topic":"rag"}}]}'
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Authorization: Bearer ${RAG_DEMO_TOKEN}" \
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
