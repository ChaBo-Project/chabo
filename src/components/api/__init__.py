"""
HTTP surfaces that are not the legacy LangServe routes.

  document_store.py  short-lived store for out-of-band document uploads
  documents.py       POST/GET/DELETE /v1/documents
  openai_compat.py   /v1/chat/completions and /v1/models
"""
from .document_store import DocumentStore, StoredDocument
from .documents import build_documents_router
from .openai_compat import build_openai_router

__all__ = [
    "DocumentStore",
    "StoredDocument",
    "build_documents_router",
    "build_openai_router",
]
