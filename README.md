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

## Prerequisites

- Python 3.11+
- Access to HuggingFace Inference Endpoints (embedding, reranking, LLM)
- Qdrant vector database instance (existing)
- Docker (optional, for containerized deployment)

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (see Configuration section)
cp params.cfg.example params.cfg
# Edit params.cfg with your endpoints

# Set required environment variables
export HF_TOKEN=your_huggingface_token
export QDRANT_API_KEY=your_qdrant_api_key

# Run the application
python src/main.py
```

### Docker

```bash
# Build and run
docker build -t chabo .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e QDRANT_API_KEY=your_key \
  chabo
```

## Configuration

### params.cfg

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

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |
| `COHERE_API_KEY` | If using Cohere | Cohere API key |

## Deployment Options

### Option 1: Backend Only (HuggingFace Spaces / Standalone)

Use the root `Dockerfile` to deploy ChaBo as a standalone API. This is the setup used on HuggingFace Spaces, where ChatUI runs as a separate Space.

```bash
docker build -t chabo .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e QDRANT_API_KEY=your_key \
  chabo
```

### Option 2: Full Stack with ChatUI (Docker Compose)

The `docker-compose/` directory contains everything needed to run both ChaBo and a [ChatUI](https://github.com/huggingface/chat-ui) frontend as a single stack. ChatUI uses a pre-built image from GHCR (`ghcr.io/m-tyrrell/chat-ui-db:0.9.4-patched`) which includes an embedded MongoDB instance.

**Setup:**

```bash
# 1. Create the ChatUI configuration from the template
cp docker-compose/chatui.env.local.template docker-compose/chatui.env.local

# 2. Edit chatui.env.local to customize app name, descriptions, etc.
#    (endpoint URLs are pre-configured for the internal Docker network)

# 3. Set your backend environment variables
export HF_TOKEN=your_huggingface_token
export QDRANT_API_KEY=your_qdrant_api_key
#  # For testing in non-HTTPS environments, it is http://138.197.179.117:3000
export UI_ORIGIN=your_ui_url
# 4. Build and start only backend services
docker-compose -f docker-compose/docker-compose.yml up --build

# or build and run backend(chabo) and qdrant (infra) and ui services
docker-compose -f docker-compose/docker-compose.yml --profile infra --profile ui up -d --build

# Or run in detached mode
docker-compose -f docker-compose/docker-compose.yml up -d --build

# View logs
docker-compose -f docker-compose/docker-compose.yml logs -f

# Stop services
docker-compose -f docker-compose/docker-compose.yml down

# To push your embedding data into the Qdrant vector database, replace the data/data.parquet with your own data
#  run the following command. 
# **Note:** Ensure you use the same collection name as defined in your `params.cfg` \
# and the correct vector dimension for your embedding model (e.g., 1024 for BGE-large).

docker exec -it docker-compose_chabo_1 python src/components/ingestor/upload_parquet.py \
    --file data/data.parquet \
    --collection YOUR_COLLECTION_NAME \
    --vector_size 1024

**Service URLs:**
- ChatUI Frontend: http://localhost:3000
- ChaBo API: http://localhost:7860
- API Documentation: http://localhost:7860/docs

**How it works:**
- The `chabo` service builds from the repo root using `backend.Dockerfile`
- The `chatui` service builds from `chatui.Dockerfile`, which pulls the pre-built ChatUI image from GHCR
- ChatUI connects to ChaBo via the internal Docker network (`http://chabo:7860`)
- MongoDB data and model directories are persisted via Docker volumes

### Testing ChatUI over HTTP (Non-HTTPS)

When deploying to a server without HTTPS (e.g., a VPS accessed via IP address), ChatUI requires two additional environment variables set on the `chatui` service in `docker-compose.yml`:

In `docker-compose.yml`:
```yaml
environment:
  - ORIGIN=http://<your-server-ip>:3000
```

In `chatui.env.local`:
```
ALLOW_INSECURE_COOKIES=true
```

- **`ORIGIN`**: Tells SvelteKit the expected origin for CSRF protection. Without this, form submissions (e.g., sending a message) return a 403 error.
- **`ALLOW_INSECURE_COOKIES`**: Allows session cookies over plain HTTP. By default, ChatUI sets cookies as `Secure` (HTTPS-only), which causes the browser to silently drop them over HTTP, resulting in 403 errors on conversation pages.

These are not needed when running behind HTTPS (e.g., HuggingFace Spaces, or behind a reverse proxy like Caddy/nginx with TLS).

## Troubleshooting: ChatUI Not Starting (Docker Compose)

If the ChatUI container starts but you cannot connect on port 3000, the most likely cause is MongoDB failing to start inside the container.

**Symptoms:**
- ChaBo backend is accessible on port 7860, but ChatUI on port 3000 is unreachable
- `docker-compose logs chatui` shows `MongoServerSelectionError: connect ECONNREFUSED 127.0.0.1:27017`

**Root cause:** The pre-built ChatUI image (`ghcr.io/m-tyrrell/chat-ui-db:0.9.4-patched`) is `linux/amd64` only. On Apple Silicon (ARM) Macs, it runs under emulation, which can cause the embedded MongoDB to fail silently.

**Diagnosis steps:**

1. Check if the chatui container is actually running:
   ```bash
   docker-compose -f docker-compose/docker-compose.yml ps
   ```

2. Check the chatui logs for MongoDB connection errors:
   ```bash
   docker-compose -f docker-compose/docker-compose.yml logs chatui
   ```

3. Look for permission errors (`chown: Operation not permitted`) — this indicates the startup script lacks root privileges to set up MongoDB data directories.

**Fixes applied:**

1. **Permission fix:** Removed `USER user` before `CMD` in `chatui.Dockerfile` so the startup script runs as root and can `chown` the volume directories and start MongoDB.

2. **Platform declaration:** Added `platform: linux/amd64` to the chatui service in `docker-compose.yml` to explicitly request amd64 emulation on ARM hosts.

**If MongoDB still fails under emulation (ARM hosts):**

A more robust solution is to run MongoDB as a separate container using the official `mongo` image (which has native ARM builds) and point ChatUI at it by changing `MONGODB_URL` in `chatui.env.local` from `mongodb://localhost:27017` to `mongodb://mongodb:27017`. This requires adding a `mongodb` service to `docker-compose.yml`.

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