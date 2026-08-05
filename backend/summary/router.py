import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import Optional

from llm_client import LLMClient
from doc_reader import extract_text
from document_store import resolve_upload

router = APIRouter()

llm = LLMClient()
print(f"[summary] service ready | flow={llm.flow}" + (f" | model={llm.model_id}" if llm.flow == "server" else ""))

LANGUAGE_NAMES = {
    'hi': 'Hindi', 'te': 'Telugu', 'mr': 'Marathi', 'kn': 'Kannada',
    'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'or': 'Odia',
    'pa': 'Punjabi', 'as': 'Assamese', 'ml': 'Malayalam', 'en': 'English',
    'rajasthani': 'Hindi',
}

LENGTH_INSTRUCTIONS = {
    "short":  "Write a concise summary in 2–3 sentences covering the main point only.",
    "medium": "Write a summary in 3–5 paragraphs covering the key points and important details.",
    "long":   "Write a detailed, comprehensive summary covering all major sections, key facts, figures, and conclusions.",
}


@router.get("/")
def root():
    return {"status": "running", "service": "ai-summarize", "version": "1.0.0"}


@router.post("/summarize")
async def summarize_document(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Body(None),
    length: str = Body("medium"),
    language: str = Body("en"),
    llm_provider: Optional[str] = Body("anthropic"),
):
    """Generate an AI summary of a document — pass either `file` or a `document_id` from a prior upload."""
    content, filename, _ = await resolve_upload(file, document_id)
    try:
        text = extract_text(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

    lang_name = LANGUAGE_NAMES.get(language, language)
    length_instr = LENGTH_INSTRUCTIONS.get(length, LENGTH_INSTRUCTIONS["medium"])

    system = (
        "You are an expert document analyst. Summarize documents clearly and accurately. "
        "Preserve important facts, figures, names, and dates. Do not add information not present in the document."
    )
    user = (
        f"Summarize the following document in {lang_name}.\n"
        f"{length_instr}\n\n"
        f"Return only the summary text — no preamble, no headings, no explanation.\n\n"
        f"Document:\n{text[:12000]}"
    )

    try:
        summary = llm.call(system, user, provider=llm_provider or "anthropic")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    return {"summary": summary, "length": length, "language": language, "provider": llm_provider}


@router.post("/summarize-sections")
async def summarize_sections(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Body(None),
    language: str = Body("en"),
    llm_provider: Optional[str] = Body("anthropic"),
):
    """Generate per-section summaries with headings. Pass either `file` or a `document_id`."""
    content, filename, _ = await resolve_upload(file, document_id)
    try:
        text = extract_text(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

    lang_name = LANGUAGE_NAMES.get(language, language)

    system = (
        "You are an expert document analyst. Identify the logical sections of a document "
        "and write a concise summary for each. Output valid JSON only."
    )
    user = (
        f"Analyse the following document and return a JSON array of section summaries in {lang_name}.\n"
        f"Each element must have: {{\"heading\": \"<section title>\", \"summary\": \"<2–3 sentence summary>\"}}\n"
        f"Return ONLY the JSON array — no markdown, no explanation.\n\n"
        f"Document:\n{text[:12000]}"
    )

    try:
        raw = llm.call(system, user, provider=llm_provider or "anthropic")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    import json, re
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        sections = json.loads(raw)
    except json.JSONDecodeError:
        sections = [{"heading": "Summary", "summary": raw}]

    return {"sections": sections, "language": language, "provider": llm_provider}
