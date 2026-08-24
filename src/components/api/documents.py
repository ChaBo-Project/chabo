"""
/v1/documents — frontend-agnostic ingestion endpoint.

2-step upload for ad hoc chat attachments — upload the file, get a `document_id`, pass the id on the next
chat request.
"""
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from components.ingestor.ingestor import process_document

from .document_store import DocumentStore

logger = logging.getLogger(__name__)


def build_documents_router(store: DocumentStore, max_upload_bytes: int) -> APIRouter:
    """
    Build the /v1/documents router bound to a document store.

    Args:
        store: where ingested uploads live until they expire.
        max_upload_bytes: reject anything larger before ingesting it.
    """
    router = APIRouter(prefix="/v1/documents", tags=["documents"])

    @router.post("")
    async def upload_document(file: UploadFile = File(...)):
        """
        Ingest one PDF/DOCX and return a short-lived document id.

        The id goes back on a chat request as `document_ids: ["doc_..."]`; the ingested text
        is then prepended to the retrieved context exactly as an inline upload would be.
        """
        filename = file.filename or "uploaded_file"
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {max_upload_bytes} byte upload limit.",
            )

        try:
            context = process_document(content, filename)
        except ValueError as e:
            # Unsupported extension — a client error, not a server fault.
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("Document ingestion failed for %s: %s", filename, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process document: {e}") from e

        if not context.strip():
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the document.",
            )

        doc = store.add(filename=filename, context=context, size_bytes=len(content))
        return doc.public_dict()

    @router.get("/{document_id}")
    async def get_document(document_id: str):
        """Metadata for a stored document (404 once it has expired or been deleted)."""
        doc = store.get(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Unknown document id: {document_id}")
        return doc.public_dict()

    @router.delete("/{document_id}")
    async def delete_document(document_id: str):
        """Drop a stored document early rather than waiting for its TTL."""
        if not store.delete(document_id):
            raise HTTPException(status_code=404, detail=f"Unknown document id: {document_id}")
        return {"id": document_id, "object": "chabo.document", "deleted": True}

    return router
