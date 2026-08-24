"""
Tests for the /v1/documents backing store (components/api/document_store.py).

*No network required (.env / services etc.)

Covers:
- round-tripping an ingested upload
- expiry by TTL, eviction at capacity, explicit deletion
- the metadata view never leaking the ingested text
"""
from components.api.document_store import DocumentStore


def test_a_stored_document_can_be_read_back_by_id():
    store = DocumentStore()
    doc = store.add(filename="report.pdf", context="[Chunk 1]: hello", size_bytes=42)
    assert store.get(doc.id).context == "[Chunk 1]: hello"


def test_ids_are_unique_per_upload():
    store = DocumentStore()
    a = store.add("a.pdf", "one", 1)
    b = store.add("b.pdf", "two", 1)
    assert a.id != b.id


def test_an_unknown_id_returns_none():
    assert DocumentStore().get("doc_missing") is None


def test_a_document_is_gone_once_its_ttl_has_passed():
    store = DocumentStore(ttl_seconds=0)
    doc = store.add("a.pdf", "text", 1)
    assert store.get(doc.id) is None


def test_the_oldest_document_is_evicted_at_capacity():
    store = DocumentStore(max_documents=2)
    first = store.add("a.pdf", "a", 1)
    second = store.add("b.pdf", "b", 1)
    third = store.add("c.pdf", "c", 1)
    assert store.get(first.id) is None
    assert store.get(second.id) is not None and store.get(third.id) is not None


def test_deleting_a_document_reports_whether_it_existed():
    store = DocumentStore()
    doc = store.add("a.pdf", "text", 1)
    assert store.delete(doc.id) is True
    assert store.delete(doc.id) is False


def test_the_metadata_view_reports_sizes_without_the_ingested_text():
    store = DocumentStore()
    doc = store.add("report.pdf", "[Chunk 1]: hello", 2048)
    public = doc.public_dict()
    assert public["filename"] == "report.pdf"
    assert public["bytes"] == 2048
    assert public["context_chars"] == len("[Chunk 1]: hello")
    assert "context" not in public
