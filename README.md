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

**Supported LLM Providers:** HuggingFace, OpenAI, Anthropic, Cohere

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
INFERENCE_PROVIDER = novita
ORGANIZATION = GIZ
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
| `UI_ORIGIN` | For non-HTTPS | ChatUI origin URL (e.g. `http://your-server-ip:3000`) |
| `COMPOSE_PROFILES` | No | Comma-separated profiles to enable (see table below) |
| `EMBEDDING_ENDPOINT_URL` | If using local TEI | `http://tei-embedding:80` |
| `RERANKER_ENDPOINT_URL` | If using local TEI | `http://tei-reranker:80` |
| `TEI_EMBEDDING_MODEL` | If using local TEI | Model ID (default: `BAAI/bge-base-en-v1.5`) |
| `TEI_RERANKER_MODEL` | If using local TEI | Model ID (default: `BAAI/bge-reranker-base`) |

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


## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.