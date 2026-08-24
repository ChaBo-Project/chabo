"""
OpenAI-compatible chat surface: POST /v1/chat/completions, GET /v1/models.

Endpoint for standardized chat frontends (OpenWebUI, LibreChat etc.) 
Uses the same graph as Chabo-ChatUI - i.e. `_consume_stream` (only the renderer differs).

Deviations from OpenAI's API:
- Sampling parameters (`temperature`, `top_p`, `max_tokens`, …) are ignored.
- `usage` is omitted. Token accounting isn't available uniformly across inference providers
- Client-supplied `system` message is ignored. The system prompt is set in Chabo `prompts.py`
- 2 non-standard request fields carry file attachments, since no chat protocol does:
   - `document_ids` (ids from POST /v1/documents)
   - inline base64 `files`
- Responses may carry a top-level `citations` array (Perplexity-style). Sources are always *also*
  rendered as markdown in the message body.
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from components.guardrails.output_classification import OutputClassificationConfig
from components.orchestration.renderers import MarkdownRenderer, OpenAIChunkRenderer
from components.orchestration.ui_adapters import (
    _consume_stream,
    _make_output_classifier,
    _make_output_filter,
    decode_base64_file,
    prepare_conversation,
    process_query_streaming,
)

from .document_store import DocumentStore

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """
    One OpenAI-shaped message.

    `content` may be a plain string or the multimodal parts array; extra keys (tool calls,
    `name`, UI-specific annotations) are ignored.
    """
    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[Dict[str, Any]], None] = None


class ChatCompletionRequest(BaseModel):
    """
    A /v1/chat/completions request. Unknown fields are accepted (see module docstring).
    """
    model_config = ConfigDict(extra="allow")

    messages: List[ChatMessage] = Field(default_factory=list)
    model: Optional[str] = None
    stream: bool = False
    #: Ids returned by POST /v1/documents, attached to this turn.
    document_ids: Optional[List[str]] = None
    #: Inline base64 attachments ({name, type: "base64", content}) for clients that can't
    #: make a separate upload call — the ChatUI file payload shape.
    files: Optional[List[Dict[str, Any]]] = None
    #: Some frontends forward extra body fields nested under `metadata`; document ids are
    #: read from there too.
    metadata: Optional[Dict[str, Any]] = None


def extract_text(content: Union[str, List[Dict[str, Any]], None]) -> str:
    """
    Flatten OpenAI message content to plain text.

    Keeps only `text` — images and other modalities are dropped, since the RAG pipeline is text-only.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(p for p in parts if p)


def _requested_document_ids(request: ChatCompletionRequest) -> List[str]:
    """Collect document ids from the top level and from a `metadata` passthrough."""
    ids = list(request.document_ids or [])
    if isinstance(request.metadata, dict):
        nested = request.metadata.get("document_ids")
        if isinstance(nested, list):
            ids.extend(str(i) for i in nested)
        elif isinstance(nested, str):
            ids.append(nested)
    return ids


def resolve_attachments(
    request: ChatCompletionRequest, store: Optional[DocumentStore]
) -> tuple[Optional[str], Optional[bytes], Optional[str]]:
    """
    Resolve this turn's attachments to (ingestor_context, file_bytes, filename).

    Stored ids arrive already ingested (that work happened at upload time); an inline base64
    file is passed through as raw bytes so `ingest_node` handles it exactly as it does for a
    ChatUI upload. 
    """
    chunks: List[str] = []
    stored_names: List[str] = []

    doc_ids = _requested_document_ids(request)
    if doc_ids and store is None:
        raise HTTPException(status_code=400, detail="Document uploads are not enabled.")
    for doc_id in doc_ids:
        doc = store.get(doc_id)
        if doc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown or expired document id: {doc_id}. Re-upload via POST /v1/documents.",
            )
        chunks.append(doc.context)
        if doc.filename:
            stored_names.append(doc.filename)

    file_content = None
    filename = None
    for file_info in request.files or []:
        if not isinstance(file_info, dict):
            continue
        try:
            file_content, filename = decode_base64_file(file_info)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if file_content:
            break  # one inline file per turn, matching the ChatUI file route

    # `filename` labels the attachment in the answer's citations (nodes.py builds the
    # ingestor Document with it, defaulting to "unknown"). An inline file must keep its own
    # name because ingest_node parses the extension from it; stored ids are already ingested,
    # so there the name is purely the citation label.
    if file_content is None and stored_names:
        filename = ", ".join(stored_names)

    return ("\n\n".join(c for c in chunks if c) or None), file_content, filename


def build_openai_router(
    compiled_graph,
    *,
    model_name: str = "chabo",
    max_turns: int = 3,
    max_chars: int = 8000,
    blocklist=None,
    blocklist_notice: str = "[response withheld]",
    classification_config: Optional[OutputClassificationConfig] = None,
    document_store: Optional[DocumentStore] = None,
    include_citations: bool = True,
) -> APIRouter:
    """
    Build the OpenAI-compatible router.

    Everything the graph needs is injected here (same pattern as the LangServe adapters), so
    the route handlers stay free of module-level state.
    """
    router = APIRouter(prefix="/v1", tags=["openai"])

    def _run_pipeline(request: ChatCompletionRequest):
        """Common request unpacking -> the internal event stream for this turn."""
        if not request.messages:
            raise HTTPException(status_code=400, detail="`messages` must not be empty.")

        # Normalise to the {role, content-as-text} shape the conversation helpers expect.
        messages = [
            {"role": m.role, "content": extract_text(m.content)} for m in request.messages
        ]
        query, conversation_context, user_messages_history = prepare_conversation(
            messages, max_turns=max_turns, max_chars=max_chars
        )
        if not (query or "").strip():
            raise HTTPException(status_code=400, detail="No user message found in `messages`.")

        ingestor_context, file_content, filename = resolve_attachments(request, document_store)

        return process_query_streaming(
            compiled_graph=compiled_graph,
            query=query,
            conversation_context=conversation_context,
            user_messages_history=user_messages_history,
            file_content=file_content,
            filename=filename,
            ingestor_context=ingestor_context,
            session_type="openai",
        )

    @router.get("/models")
    async def list_models():
        """
        Model listing — necessary for OpenWebUI/LibreChat to initiate chat
        - They use it to populate their model selector
        - Has no bearing on the actual LLM used (defined in config)
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "chabo",
                }
            ],
        }

    @router.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completion, streamed as SSE when `stream: true`, otherwise a single JSON body."""
        response_model = request.model or model_name
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        process_iter = _run_pipeline(request)

        output_filter = _make_output_filter(blocklist, blocklist_notice)
        classifier = _make_output_classifier(classification_config)

        if request.stream:
            renderer = OpenAIChunkRenderer(
                model=response_model,
                request_id=request_id,
                include_citations=include_citations,
            )

            async def event_stream():
                try:
                    async for frame in _consume_stream(
                        process_iter, output_filter, classifier, renderer
                    ):
                        yield frame
                except Exception as e:  # transport-level failure after headers were sent
                    logger.error("OpenAI stream failed: %s", e, exc_info=True)
                    yield renderer.error(str(e))
                    yield renderer.finish()

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # don't let a proxy buffer the stream
                },
            )

        # Non-streaming: same pipeline, same renderer output, collected into one message.
        renderer = MarkdownRenderer(trailing_flush_delay=0.0)
        parts = []
        async for piece in _consume_stream(process_iter, output_filter, classifier, renderer):
            parts.append(piece)
        content = "".join(parts)

        response: Dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        if include_citations and renderer.citations:
            response["citations"] = renderer.citations
        return response

    return router
