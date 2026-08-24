"""
Short-lived store for in-query document uploades (POST /v1/documents).

Rationale: chat frontends don't seem to natively pass file uploads (rather they process themselves and send as text).
OpenWebUI and LibreChat both run their own file RAG and never forward the upload, so file upload requires per-UI customization.
- The UI uploads to the Chabo doc store (temporary in-memory dict object)
- We pass back a `document_id` in the HTTP response
- The UI passes the id back on subsequent turns and the extracted text is pulled from the doc store

NOTE: Need to review the benefit of this approach. Not clear to me that it offers any advantage other than for Chabo-ChatUI
i.e. if the other UIs are already processing files and submitting only text, then this does nothing.

"""
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_DOCUMENTS = 100


@dataclass
class StoredDocument:
    """One ingested upload, as returned to the client and consumed by the chat endpoint."""
    id: str
    filename: str
    context: str
    size_bytes: int
    created_at: float
    expires_at: float

    def public_dict(self) -> Dict:
        """Metadata view — the ingested text itself is never returned to the client."""
        return {
            "id": self.id,
            "object": "chabo.document",
            "filename": self.filename,
            "bytes": self.size_bytes,
            "context_chars": len(self.context),
            "created": int(self.created_at),
            "expires_at": int(self.expires_at),
        }


class DocumentStore:
    """
    TTL + capacity bounded map of document_id -> StoredDocument.

    Bounded on both axes so a long-running deployment can't accumulate ingested text
    indefinitely: expired entries are purged on every access, and once `max_documents` is
    reached the oldest entry is evicted.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_documents: int = DEFAULT_MAX_DOCUMENTS,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_documents = max_documents
        self._docs: Dict[str, StoredDocument] = {}
        self._lock = threading.Lock()

    def add(self, filename: str, context: str, size_bytes: int) -> StoredDocument:
        now = time.time()
        doc = StoredDocument(
            id=f"doc_{uuid.uuid4().hex}",
            filename=filename,
            context=context,
            size_bytes=size_bytes,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._purge_locked(now)
            while len(self._docs) >= self.max_documents:
                oldest = min(self._docs.values(), key=lambda d: d.created_at)
                self._docs.pop(oldest.id, None)
                logger.info("Document store at capacity — evicted %s", oldest.id)
            self._docs[doc.id] = doc
        logger.info(
            "Stored document %s (%s, %d chars of context)", doc.id, filename, len(context)
        )
        return doc

    def get(self, doc_id: str) -> Optional[StoredDocument]:
        now = time.time()
        with self._lock:
            self._purge_locked(now)
            return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            return self._docs.pop(doc_id, None) is not None

    def _purge_locked(self, now: float) -> None:
        expired = [d_id for d_id, doc in self._docs.items() if doc.expires_at <= now]
        for d_id in expired:
            del self._docs[d_id]
        if expired:
            logger.info("Purged %d expired document(s) from the store", len(expired))
