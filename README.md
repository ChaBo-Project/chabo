# ChaBo

A RAG (Retrieval-Augmented Generation) orchestrator API built with FastAPI, LangChain, and LangGraph. ChaBo orchestrates embedding, vector search, reranking, and LLM generation to answer queries using retrieved context.

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────────────┐
│   ChatUI    │────▶│                    ChaBo                        │
│  (Frontend) │     │  ┌─────────┐   ┌─────────┐   ┌───────────────┐  │
└─────────────┘     │  │ Embed   │──▶│ Search  │──▶│   Rerank      │  │
                    │  │ (HF)    │   │ (Qdrant)│   │   (HF)        │  │
                    │  └─────────┘   └─────────┘   └───────┬───────┘  │
                    │                                      │          │
                    │                              ┌───────▼───────┐  │
                    │                              │   Generate    │  │
                    │                              │ (Multi-LLM)   │  │
                    │                              └───────────────┘  │
                    └─────────────────────────────────────────────────┘
```

**Pipeline:** Query → Embed → Vector Search → Rerank → Generate (with citations)

**Supported LLM Providers:** HuggingFace, OpenAI, Anthropic, Cohere, Azure OpenAI

## Deployment

### Option 1: Backend Only (HuggingFace Spaces / Standalone)

Deploy ChaBo as a standalone API using the root `Dockerfile`. This is the setup used on HuggingFace Spaces, where a frontend (e.g. ChatUI) runs separately.

**Prerequisites:** Remote HuggingFace Inference Endpoints for embedding and reranking, an existing Qdrant instance, and API keys.

#### Configuration

Edit `params.cfg` with your service endpoints:

```ini
[hf_endpoints]
embedding_endpoint_url = https://your-embedding-endpoint.huggingface.cloud
reranker_endpoint_url = https://your-reranker-endpoint.huggingface.cloud

[qdrant]
mode = native
url = https://your-qdrant-instance.cloud.qdrant.io:6333
collection = your_collection

[retrieval]
initial_k = 20
final_k = 5

[generator]
PROVIDER = huggingface
MODEL = meta-llama/Meta-Llama-3-8B-Instruct
MAX_TOKENS = 1024
TEMPERATURE = 0.1
INFERENCE_PROVIDER = ABC
ORGANIZATION = XYZ
CONTEXT_META_FIELDS = filename,project_id,document_source
TITLE_META_FIELDS = filename,page
```

Pass API keys as environment variables at runtime:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |
| `COHERE_API_KEY` | If using Cohere | Cohere API key |
| `AZURE_API_KEY` | If using Azure OpenAI | Azure OpenAI API key |

#### Build and Run

```bash
docker build -t chabo .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e QDRANT_API_KEY=your_key \
  chabo
```

---

### Option 2: Full Stack with Docker Compose

Run ChaBo with a [ChatUI](https://github.com/huggingface/chat-ui) frontend as a single stack. Docker Compose profiles optionally add local embedding/reranking (TEI) and a local Qdrant instance — so you can go fully self-contained with no external dependencies beyond an LLM provider.

#### Configuration

Configuration is split across three files:

- **`params.cfg`** (repo root) — Qdrant connection, retrieval parameters, generator/LLM settings, and ingestor chunking config. This is the same file used in Option 1, but when using local TEI or local Qdrant, the endpoint URLs in `.env` override what's in `params.cfg`.
- **`docker-compose/.env`** — API keys, Compose profiles, and endpoint overrides for local TEI containers.
- **`docker-compose/chatui.env.local`** — ChatUI frontend settings (app name, model endpoints, UI options).

Set up the Docker Compose files:

**1. Environment file** — controls backend services and Compose profiles:

```bash
cp docker-compose/.env.example docker-compose/.env
```

Edit `docker-compose/.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `COMPOSE_PROFILES` | No | Comma-separated profiles to enable (see table below) |
| `TEI_EMBEDDING_MODEL` | If using local TEI | Model ID (default: `BAAI/bge-base-en-v1.5`) |
| `TEI_RERANKER_MODEL` | If using local TEI | Model ID (default: `BAAI/bge-reranker-base`) |

Embedding and reranker endpoint URLs are configured in `params.cfg` under `[hf_endpoints]`. When using local TEI, set them to `http://tei-embedding:80` and `http://tei-reranker:80`.

**2. ChatUI config** — controls the frontend app name, model endpoints, and UI settings:

```bash
cp docker-compose/chatui.env.local.template docker-compose/chatui.env.local
# Edit chatui.env.local to customize (endpoint URLs are pre-configured for the Docker network)
```

#### Compose Profiles

Profiles add optional services on top of the base stack (ChaBo + ChatUI):

| Profile | Services added | Use case |
|---------|---------------|----------|
| `local` | `tei-embedding` (port 8081), `tei-reranker` (port 8082) | Self-hosted embedding and reranking instead of remote HF endpoints |
| `infra` | `qdrant` (port 6333) | Local Qdrant instance instead of a remote one |

Set profiles in `.env` (e.g. `COMPOSE_PROFILES=local,infra`) or via the CLI.

> **Important:** When using local TEI, the embedding model (`TEI_EMBEDDING_MODEL`) must match the model used to create your Qdrant collection. Mismatched models will produce poor search results.

#### Build and Run

```bash
# Start base stack (ChaBo + ChatUI, using remote HF endpoints and remote Qdrant)
docker-compose -f docker-compose/docker-compose.yml up --build

# Start with local TEI embedding/reranking
COMPOSE_PROFILES=local docker-compose -f docker-compose/docker-compose.yml up --build

# Start fully self-contained (local TEI + local Qdrant)
COMPOSE_PROFILES=local,infra docker-compose -f docker-compose/docker-compose.yml up --build
# or docker-compose -f docker-compose/docker-compose.yml --profile infra --profile local up -d --build

# Run in detached mode
docker-compose -f docker-compose/docker-compose.yml up -d --build

# View logs
docker-compose -f docker-compose/docker-compose.yml logs -f

# Stop services
docker-compose -f docker-compose/docker-compose.yml down
```

**First run with local TEI:** The TEI containers download models on first startup (1-3 minutes depending on model size). Models are cached in Docker volumes, so subsequent starts are fast.

**GPU support:** The default TEI images are CPU-only. For GPU acceleration, edit `docker-compose/docker-compose.yml` and swap in the GPU image variant (see comments in the file).

#### Data Upload

To populate a Qdrant collection with your embedding data, place your data file at `data/data.parquet` and run:

```bash
docker exec -it docker-compose_chabo_1 python src/components/ingestor/upload_parquet.py \
    --file data/data.parquet \
    --collection YOUR_COLLECTION_NAME \
    --vector_size 1024
```

> **Note:** Use the same collection name as in your `params.cfg` and the correct vector dimension for your embedding model (e.g. 1024 for BGE-large, 768 for BGE-base).

#### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| ChatUI | http://localhost:3000 | Chat frontend |
| ChaBo API | http://localhost:7860 | Backend API |
| API Docs | http://localhost:7860/docs | Interactive documentation |

## Non-HTTPS Deployments

When deploying to a server without HTTPS (e.g. a VPS accessed via IP address), ChatUI needs two settings to avoid 403 errors:

1. Set `UI_ORIGIN` in `docker-compose/.env` to your server's URL (e.g. `http://your-server-ip:3000`). This tells SvelteKit the expected origin for CSRF protection.
2. Uncomment `ALLOW_INSECURE_COOKIES=true` in `docker-compose/chatui.env.local`. This allows session cookies over plain HTTP.

Not needed behind HTTPS (e.g. HuggingFace Spaces, or behind a reverse proxy with TLS).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/chatfed-ui-stream` | POST | Text query streaming (LangServe) |
| `/chatfed-with-file-stream` | POST | File upload + query streaming (LangServe) |


## Health Checks & Testing

The `tests/health/` directory contains scripts to verify your setup before or after deployment. These run **locally outside Docker**, against already-running services, using the project's virtual environment.

### Prerequisites

Before running any health checks ensure:
1. Services are up — Qdrant, embedding endpoint, reranker endpoint
2. `params.cfg` is configured with correct endpoint URLs and collection name
3. Required env vars are exported (`HF_TOKEN`, `QDRANT_API_KEY`, etc.)

### Setup

```bash
# Navigate to repo root — required for src/ imports to resolve
cd /path/to/chabo

```

### Running the full health suite

```bash
python tests/health/run_all.py
```

`run_all.py` checks each component in order and prints a pass/fail summary:

| Step | Check | What it verifies |
|------|-------|-----------------|
| 1 | Qdrant | Reachable and configured collection exists |
| 1 | Embedding endpoint | Returns a valid vector |
| 1 | Reranker endpoint | Returns scores |
| 2 | Retriever unit | Full retrieval + reranking returns ranked documents |
| 2 | RAG pipeline | End-to-end retrieval → streaming generation with a sample query |

Step 2 (component checks) only runs if all Step 1 connectivity checks pass. Logs are written to `tests/health/logs/` with a timestamp for each run.

### Running individual scripts

```bash
# Retriever + pipeline tests only (skips connectivity pre-checks)
python tests/health/test_rag_pipeline.py
```

Edit the `test_cases` list in `test_rag_pipeline.py` to add your own queries.

---

## RAG Evaluation

The `tests/eval/` directory contains scripts to evaluate retrieval and reranking quality using an LLM-as-judge approach. Like health checks, these run locally outside Docker with the venv active from the repo root.

### How it works

Evaluation runs in two stages:

**Stage 1 — Retrieval** (`run_retrieval_only`)
Runs each question through the full retriever pipeline, capturing both raw vector search candidates and final reranked results. Output saved to `tests/eval/results/retrieval_eval_results.json`.

**Stage 2 — LLM Judging** (`run_evaluation_batch`)
Loads the retrieval results and uses the configured LLM to judge each retrieved document for relevance. Saves a judged report to `tests/eval/results/judged_eval_report.json`. Supports **checkpointing** — if interrupted, it resumes from where it left off.

### Setup

```bash
cd /path/to/chabo
source chabo_env/bin/activate
```

### Define your test questions

Edit `tests/eval/test_questions.py` to add your questions:

```python
questions = [
    "Your question here",
    "Another question",
]
```

### Run evaluation

Pass the `--mode` flag to select which stage to run:

```bash
# Step 1: Run retrieval and save results
python tests/eval/eval.py --mode retrieval

# Step 2: Judge retrieved results with LLM (resumes from checkpoint if interrupted)
python tests/eval/eval.py --mode batch

# Quick sample run (first 2 questions only, useful for testing)
python tests/eval/eval.py --mode sample
```

`--mode retrieval` is the default if no flag is provided.

Results are saved to `tests/eval/results/` (gitignored).

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
