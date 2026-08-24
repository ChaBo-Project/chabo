# Connecting a frontend to ChaBo

ChaBo exposes two HTTP surfaces:

| Surface | Endpoints | Who it's for |
|---|---|---|
| **OpenAI-compatible** | `/v1/chat/completions`, `/v1/models`, `/v1/documents` | Any modern frontend (OpenWebUI, LibreChat, an SDK, curl) — no bespoke connector |
| **LangServe** | `/chatfed-ui-stream`, `/chatfed-with-file-stream` | The existing Chabo-ChatUI, whose "langserve-streaming" connector understands LangServe's own envelope and nothing else |

Both run the same graph, the same guardrails, and the same citation rendering. They differ
only in the *renderer* that turns ChaBo's internal event stream into wire format
(`src/components/orchestration/renderers.py`). New frontends should use the OpenAI surface;
the LangServe routes are kept so the current ChatUI deployments don't break.

## `/v1/chat/completions`

Standard OpenAI request and response shapes, streaming (`stream: true`, SSE of
`chat.completion.chunk`) and non-streaming.

```bash
curl -N http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chabo","stream":true,
       "messages":[{"role":"user","content":"When is wheat sown?"}]}'
```

Deviations from OpenAI's API:
- Sampling parameters (`temperature`, `top_p`, `max_tokens`, …) are ignored.
- `usage` is omitted. Token accounting isn't available uniformly across inference providers
- Client-supplied `system` message is ignored. The system prompt is set in Chabo `prompts.py`
- 2 non-standard request fields carry file attachments, since no chat protocol does:
   - `document_ids` (ids from POST /v1/documents)
   - inline base64 `files`
- Responses may carry a top-level `citations` array (Perplexity-style). Sources are always *also*
  rendered as markdown in the message body.

`GET /v1/models` advertises a single model id (`[api] model_name`) — this is how OpenWebUI
and LibreChat discover the endpoint. It names the deployment, not the LLM.

## `/v1/documents` — the file bridge

Neither OpenWebUI nor LibreChat forwards raw uploaded files to a custom backend: both run
their own file RAG. File delivery is therefore a per-UI bridge, not a protocol guarantee.
ChaBo's side of that bridge is a two-step upload:

```bash
# 1. upload — returns {"id": "doc_...", ...}
curl -F 'file=@report.pdf' http://localhost:7860/v1/documents

# 2. reference it on the chat call
curl -N http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chabo","stream":true,"document_ids":["doc_..."],
       "messages":[{"role":"user","content":"Summarise the attached report."}]}'
```

- Accepts `.pdf`, `.docx`, `.txt`, `.md`. The upload is ingested once (same
  `process_document()` the graph uses) and the resulting chunk text is held in memory until
  its TTL expires; only the id travels with the chat request.
- An unknown or expired id **fails the request with 404** rather than quietly answering
  without the attachment — a silently ignored upload is the worse failure.
- `GET /v1/documents/{id}` returns metadata; `DELETE /v1/documents/{id}` drops it early.
- The store is per-process and in-memory. Multiple workers behind a load balancer need
  sticky sessions (or a shared store) for an id to be found again on the chat call.
- Bounds live in `[api]`: `document_ttl_seconds`, `max_documents`, `max_upload_bytes`.

Clients that can't make a second call can inline the file instead:
`{"files": [{"name": "report.pdf", "type": "base64", "content": "<base64>"}]}`. These bytes
go through `ingest_node` exactly as a ChatUI upload does.

## Per-UI notes

### OpenWebUI

1. Settings → Connections → add an OpenAI-compatible connection pointing at
   `http://<chabo-host>:7860/v1` (any non-empty API key). The model from `/v1/models`
   appears in the picker; chat and corpus retrieval work with no further setup.
2. For uploads, install the Pipe Function in
   [`docs/integrations/openwebui_pipe.py`](integrations/openwebui_pipe.py) (Workspace →
   Functions → paste → enable → set valves). It uploads attachments to `/v1/documents` and
   forwards the ids.
3. In Admin Settings → Interface, point the **task model** (title / tag / search-query
   generation) at a plain LLM rather than ChaBo. Those background calls arrive as ordinary
   chat completions, so leaving them on ChaBo runs a full retrieval per generated title.

**The Pipe's file path is not yet verified against a live OpenWebUI.** What a Pipe can see of
an upload is unresolved upstream ([#19963](https://github.com/open-webui/open-webui/issues/19963),
[#17293](https://github.com/open-webui/open-webui/issues/17293)), so the Pipe tries the
original bytes via OpenWebUI's files API, then OpenWebUI's own extracted text, then degrades
to a corpus-only answer. Turning on its `DEBUG_FILES` valve logs what actually arrives —
that log is the spike; prune the dead routes once it's known.

### LibreChat

Add ChaBo as a [custom endpoint](https://www.librechat.ai/docs/quick_start/custom_endpoints)
with `baseURL: http://<chabo-host>:7860/v1`. Chat, streaming, and corpus retrieval work.
Files route through LibreChat's own [RAG API](https://www.librechat.ai/docs/features/rag_api)
service, which is a separate component from a custom endpoint — either point that service at
a ChaBo-backed implementation, or accept LibreChat's native file RAG for that deployment
(only ad hoc uploaded-doc handling is affected).

### Chabo-ChatUI

Works unchanged on the LangServe routes today but can be migrated to `/v1/chat/completions`. 
Three ChatUI-shaped workarounds can then be deprecated:

1. The `langserve-streaming` connector.
2. The `doc://#` placeholder link. The fork's `langserve-streaming` endpoint finds sources by
  running `/\[([^\]]+)\]\(((?:doc|https?):\/\/[^)]+)\)/g` over any chunk containing
  `**Sources:**`, and *also* truncates the displayed text at that marker — so a source whose
  link carries no accepted scheme is dropped from the sources panel **and** from the message
  body. Hence the placeholder. 
3. `TRAILING_FLUSH_DELAY` (the sleep that keeps ChatUI's parser from dropping coalesced
  trailing chunks). The OpenAI renderer's frames are self-delimiting, so it is not required.


## Adding another frontend

Subclass `BaseRenderer` in `src/components/orchestration/renderers.py` and pass an instance
to `_consume_stream`. Everything else — graph, guards, footnote, citation logic — is shared:
`_consume_stream` is the only consumer of the internal event stream, so a new frontend never
forks the streaming logic.
