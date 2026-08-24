"""
Tests for the /v1/documents ingestion endpoint (components/api/documents.py).

*No network required (.env / services etc.) — real ingestion runs, on plain-text uploads.
Run from the repo root: the ingestor reads chunking settings from params.cfg.

Covers:
- upload -> id -> metadata -> delete lifecycle
- rejections that must not read as server faults (unsupported type, empty, oversized)
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from components.api import DocumentStore, build_documents_router

MAX_UPLOAD = 1024 * 1024


def build_client(store=None, max_upload_bytes=MAX_UPLOAD):
    store = store or DocumentStore()
    app = FastAPI()
    app.include_router(build_documents_router(store, max_upload_bytes=max_upload_bytes))
    return TestClient(app), store


def upload(client, name="notes.md", content=b"# Wheat\nSow in November."):
    return client.post("/v1/documents", files={"file": (name, content, "text/markdown")})


def test_an_upload_is_ingested_and_returns_a_document_id():
    client, store = build_client()
    body = upload(client).json()
    assert body["id"].startswith("doc_")
    assert body["filename"] == "notes.md"
    # The stored context is the ingested chunk text the graph will be seeded with.
    assert "Sow in November." in store.get(body["id"]).context


def test_an_uploaded_document_can_be_looked_up_then_deleted():
    client, _ = build_client()
    doc_id = upload(client).json()["id"]
    assert client.get(f"/v1/documents/{doc_id}").status_code == 200
    assert client.delete(f"/v1/documents/{doc_id}").json()["deleted"] is True
    assert client.get(f"/v1/documents/{doc_id}").status_code == 404


def test_deleting_an_unknown_document_is_a_404():
    client, _ = build_client()
    assert client.delete("/v1/documents/doc_missing").status_code == 404


def test_an_unsupported_file_type_is_a_client_error_not_a_server_fault():
    client, _ = build_client()
    response = client.post(
        "/v1/documents", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]  # the message names what is supported


def test_an_empty_upload_is_rejected():
    client, _ = build_client()
    assert upload(client, content=b"").status_code == 400


def test_an_oversized_upload_is_rejected_before_ingestion():
    client, _ = build_client(max_upload_bytes=16)
    assert upload(client, content=b"x" * 17).status_code == 413


def test_a_document_with_no_extractable_text_is_rejected():
    client, _ = build_client()
    assert upload(client, content=b"   \n  ").status_code == 422
