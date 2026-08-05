"""Shared document store used across feature routers within one process.

`store` is a module-level singleton. Every feature package carries an
identical copy of this file (same convention as llm_client.py/doc_reader.py)
so each stays self-contained when run standalone, while in the unified
aiService process Python's module cache makes them resolve to the same
object — giving one shared in-memory store across all features.

To move to a database later, implement `DocumentStore` against the DB and
swap the `store = InMemoryDocumentStore()` line below — callers never change.
"""

import time
import uuid
from typing import Optional, TypedDict

from fastapi import HTTPException, UploadFile


class StoredDocument(TypedDict):
    filename: str
    content_type: str
    content: bytes
    text: str
    uploaded_at: float


class DocumentStore:
    def put(self, filename: str, content_type: str, content: bytes, text: str) -> str:
        raise NotImplementedError

    def get(self, document_id: str) -> Optional[StoredDocument]:
        raise NotImplementedError

    def delete(self, document_id: str) -> bool:
        raise NotImplementedError


class InMemoryDocumentStore(DocumentStore):
    def __init__(self):
        self._docs: dict[str, StoredDocument] = {}

    def put(self, filename: str, content_type: str, content: bytes, text: str) -> str:
        document_id = str(uuid.uuid4())
        self._docs[document_id] = {
            "filename": filename,
            "content_type": content_type or "application/octet-stream",
            "content": content,
            "text": text,
            "uploaded_at": time.time(),
        }
        return document_id

    def get(self, document_id: str) -> Optional[StoredDocument]:
        return self._docs.get(document_id)

    def delete(self, document_id: str) -> bool:
        return self._docs.pop(document_id, None) is not None


store: DocumentStore = InMemoryDocumentStore()


async def resolve_upload(file: Optional[UploadFile], document_id: Optional[str]):
    """Resolve an endpoint's input to (content, filename, content_type) from
    either a fresh upload or a previously stored document_id."""
    if file is not None:
        content = await file.read()
        return content, file.filename, file.content_type or "application/octet-stream"
    if document_id:
        doc = store.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found. Upload it first.")
        return doc["content"], doc["filename"], doc["content_type"]
    raise HTTPException(status_code=400, detail="Provide either a file or a document_id.")
