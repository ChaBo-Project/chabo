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

## ChatUI Integration

ChaBo is designed to work with a modified version of [HuggingFace's ChatUI](https://github.com/huggingface/chat-ui) as its frontend. The ChatUI image is available from GHCR: (URL TO BE ADDED)

### ChatUI Configuration

Create a `.env.local` file for ChatUI with the following configuration:

```bash
MODELS=`[
  {
    "name": "ChaBo",
    "displayName": "ChaBo RAG Assistant",
    "description": "Retrieval-augmented generation powered by ChaBo.",
    "multimodal": true,
    "multimodalAcceptedMimetypes": ["application/geojson"],
    "chatPromptTemplate": "{{#each messages}}{{#ifUser}}{{content}}{{/ifUser}}{{#ifAssistant}}{{content}}{{/ifAssistant}}{{/each}}",
    "parameters": {
      "temperature": 0.0,
      "max_new_tokens": 2048
    },
    "endpoints": [{
      "type": "langserve-streaming",
      "url": "http://chabo:7860/chatfed-ui-stream",
      "streamingFileUploadUrl": "http://chabo:7860/chatfed-with-file-stream",
      "inputKey": "text",
      "fileInputKey": "files"
    }]
  }
]`

MONGODB_URL=mongodb://localhost:27017
LLM_SUMMARIZATION=false
```

### Running with Docker Compose (Not possible on huggingface spaces)

Create a `docker-compose.yml`:

```yaml
services:
  chabo:
    build: .
    ports:
      - "7860:7860"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - QDRANT_API_KEY=${QDRANT_API_KEY}

  chatui:
    image: ghcr.io/your-org/chatui:latest
    ports:
      - "3000:3000"
    env_file:
      - .env.local
    depends_on:
      - chabo
```

```bash
docker-compose up --build
```

**Service URLs:**
- ChatUI Frontend: http://localhost:3000
- ChaBo API: http://localhost:7860
- API Documentation: http://localhost:7860/docs

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