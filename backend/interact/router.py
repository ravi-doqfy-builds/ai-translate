import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from llm_client import LLMClient
from doc_reader import extract_text
from document_store import store as doc_store

router = APIRouter()

llm = LLMClient()
print(f"[interact] service ready | flow={llm.flow}" + (f" | model={llm.model_id}" if llm.flow == "server" else ""))


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    document_id: str
    question: str
    history: List[ChatMessage] = []
    llm_provider: Optional[str] = "gemini"


@router.get("/")
def root():
    return {"status": "running", "service": "ai-interact", "version": "1.0.0"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a document for Q&A. Returns a document_id usable across features."""
    content = await file.read()
    try:
        text = extract_text(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

    doc_id = doc_store.put(file.filename, file.content_type, content, text)
    print(f"[Interact] Stored doc_id={doc_id}, filename={file.filename}, chars={len(text)}")
    return {"document_id": doc_id, "filename": file.filename, "char_count": len(text)}


@router.post("/chat")
async def chat_with_document(request: ChatRequest):
    """Ask a question about a previously uploaded document."""
    doc = doc_store.get(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{request.document_id}' not found. Upload it first.")

    # Build conversation history string
    history_str = ""
    if request.history:
        lines = []
        for msg in request.history[-6:]:  # last 6 turns for context
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        history_str = "\n".join(lines) + "\n"

    system = (
        "You are an expert document analyst. Answer questions based ONLY on the provided document. "
        "If the answer is not in the document, say so clearly. Be concise and accurate."
    )
    user = (
        f"Document:\n{doc['text'][:10000]}\n\n"
        f"{history_str}"
        f"User: {request.question}\nAssistant:"
    )

    try:
        answer = llm.call(system, user, provider=request.llm_provider or "gemini")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    return {
        "answer": answer.strip(),
        "document_id": request.document_id,
        "provider": request.llm_provider,
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response as Server-Sent Events."""
    doc = doc_store.get(request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{request.document_id}' not found.")

    history_str = ""
    if request.history:
        lines = []
        for msg in request.history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        history_str = "\n".join(lines) + "\n"

    system = (
        "You are an expert document analyst. Answer questions based ONLY on the provided document. "
        "If the answer is not in the document, say so clearly. Be concise and accurate."
    )
    user = (
        f"Document:\n{doc['text'][:10000]}\n\n"
        f"{history_str}"
        f"User: {request.question}\nAssistant:"
    )

    def generate():
        try:
            for token in llm.stream(system, user, provider=request.llm_provider or "gemini"):
                print(f"[Stream] token: {repr(token[:20])}")
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            print(f"[Stream] ERROR: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Remove an ingested document from the store."""
    if not doc_store.delete(document_id):
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return {"deleted": document_id}
