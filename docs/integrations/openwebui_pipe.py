"""
title: ChaBo RAG
author: ChaBo
version: 0.1.0
license: Apache-2.0
description: Routes OpenWebUI chats to a ChaBo RAG orchestrator, bridging file uploads to /v1/documents.
requirements: aiohttp
"""
# ---------------------------------------------------------------------------------------
# This file does NOT run inside ChaBo. It is pasted into OpenWebUI (Workspace → Functions →
# "+" → paste → save, then enable it and set the valves). It lives in this repo so the
# bridge is versioned alongside the API it talks to.
#
# What it does:
#   chat      -> POST {CHABO_URL}/v1/chat/completions (streamed, SSE)
#   uploads   -> POST {CHABO_URL}/v1/documents, then passes the returned ids on the chat call
#
# Why a bridge is needed at all: OpenWebUI does its own file RAG and does not forward
# uploaded files to a custom backend over the chat call (open-webui#17293). So the file path
# has to be reconstructed here.
#
# STATUS — the chat path is straightforward; the file path is NOT yet verified against a
# live OpenWebUI (the "spike" in docs/ui-generalization-plan.md). What a Pipe can actually
# see of an upload is unresolved upstream (open-webui#19963), so `_collect_documents` tries
# the plausible routes in order and degrades quietly:
#
#   1. Fetch the ORIGINAL file bytes from OpenWebUI's own files API and upload those
#      (best fidelity — ChaBo extracts and chunks the real PDF/DOCX itself).
#   2. Fall back to the text OpenWebUI already extracted, uploaded as .txt.
#   3. Fall back to no attachment — chat and corpus retrieval work as normal.
#
# Set DEBUG_FILES to log the shape of what actually arrives; that log IS the spike. Once the
# real shape is known, delete the routes that don't apply.
# ---------------------------------------------------------------------------------------
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        CHABO_URL: str = Field(
            default="http://chabo:7860",
            description="Base URL of the ChaBo orchestrator (no trailing slash).",
        )
        MODEL_ID: str = Field(
            default="chabo",
            description="Model id to request — must match [api] model_name in params.cfg.",
        )
        MODEL_NAME: str = Field(
            default="ChaBo RAG",
            description="Display name shown in the OpenWebUI model picker.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=300, description="Seconds to wait for a full answer."
        )
        FORWARD_UPLOADS: bool = Field(
            default=True,
            description="Bridge uploaded files to ChaBo's /v1/documents endpoint.",
        )
        OPENWEBUI_URL: str = Field(
            default="http://localhost:8080",
            description="OpenWebUI's own base URL, used to fetch original uploaded files.",
        )
        DEBUG_FILES: bool = Field(
            default=False,
            description="Log the structure of incoming file payloads (use this to run the spike).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> List[dict]:
        """One entry: this ChaBo deployment."""
        return [{"id": self.valves.MODEL_ID, "name": self.valves.MODEL_NAME}]

    # -- file bridge ---------------------------------------------------------------------

    async def _upload(self, session, filename: str, content: bytes, content_type: str) -> Optional[str]:
        """Push one file to ChaBo's /v1/documents and return its document id."""
        form = aiohttp.FormData()
        form.add_field("file", content, filename=filename, content_type=content_type)
        async with session.post(f"{self.valves.CHABO_URL}/v1/documents", data=form) as resp:
            if resp.status != 200:
                logger.warning("ChaBo rejected upload %s: %s %s", filename, resp.status, await resp.text())
                return None
            return (await resp.json())["id"]

    async def _fetch_original(self, session, file_id: str, auth: Optional[str]) -> Optional[bytes]:
        """Ask OpenWebUI for the bytes of a file the user uploaded to it."""
        headers = {"Authorization": auth} if auth else {}
        url = f"{self.valves.OPENWEBUI_URL}/api/v1/files/{file_id}/content"
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    logger.info("Could not fetch original file %s: HTTP %s", file_id, resp.status)
                    return None
                return await resp.read()
        except Exception as e:  # network/permission problems must not break the chat
            logger.info("Could not fetch original file %s: %s", file_id, e)
            return None

    async def _collect_documents(self, session, body: Dict[str, Any], auth: Optional[str]) -> List[str]:
        """
        Turn whatever OpenWebUI gives us about the attachments into ChaBo document ids.

        Never raises: an attachment that cannot be bridged degrades to a normal (corpus-only)
        answer rather than a failed chat.
        """
        files = body.get("files") or []
        if self.valves.DEBUG_FILES:
            logger.info("ChaBo pipe — incoming file payload: %s", json.dumps(files, default=str)[:4000])
        if not files:
            return []

        document_ids: List[str] = []
        for entry in files:
            payload = entry.get("file", entry) if isinstance(entry, dict) else {}
            file_id = payload.get("id")
            filename = payload.get("filename") or payload.get("name") or "upload"

            # Route 1: the original bytes, extracted by ChaBo itself.
            if file_id:
                content = await self._fetch_original(session, file_id, auth)
                if content:
                    doc_id = await self._upload(session, filename, content, "application/octet-stream")
                    if doc_id:
                        document_ids.append(doc_id)
                        continue

            # Route 2: the text OpenWebUI already extracted.
            data = payload.get("data") or {}
            extracted = data.get("content") if isinstance(data, dict) else None
            if extracted:
                doc_id = await self._upload(
                    session, f"{filename}.txt", extracted.encode("utf-8"), "text/plain"
                )
                if doc_id:
                    document_ids.append(doc_id)
                    continue

            logger.info("ChaBo pipe: no usable content for attachment %s — skipping", filename)

        return document_ids

    # -- chat ----------------------------------------------------------------------------

    async def pipe(self, body: Dict[str, Any], __request__: Any = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream one ChaBo answer back to OpenWebUI."""
        auth = None
        try:
            auth = __request__.headers.get("authorization") if __request__ is not None else None
        except Exception:
            auth = None

        payload: Dict[str, Any] = {
            "model": self.valves.MODEL_ID,
            "messages": body.get("messages", []),
            "stream": True,
        }

        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.valves.FORWARD_UPLOADS:
                document_ids = await self._collect_documents(session, body, auth)
                if document_ids:
                    payload["document_ids"] = document_ids

            async with session.post(
                f"{self.valves.CHABO_URL}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    yield f"Error: ChaBo returned HTTP {resp.status} — {await resp.text()}"
                    return

                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
