import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import re
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from typing import Optional

from llm_client import LLMClient
from doc_reader import extract_text
from document_store import resolve_upload

router = APIRouter()

llm = LLMClient()
print(f"[comparison] service ready | flow={llm.flow}" + (f" | model={llm.model_id}" if llm.flow == "server" else ""))

LANGUAGE_NAMES = {
    'hi': 'Hindi', 'te': 'Telugu', 'mr': 'Marathi', 'kn': 'Kannada',
    'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'en': 'English',
}


def _text_diff(text_a: str, text_b: str) -> dict:
    """Simple line-level diff."""
    lines_a = set(text_a.splitlines())
    lines_b = set(text_b.splitlines())
    only_in_a = [l for l in lines_a - lines_b if l.strip()]
    only_in_b = [l for l in lines_b - lines_a if l.strip()]
    common = len(lines_a & lines_b)
    return {
        "only_in_a": only_in_a[:50],
        "only_in_b": only_in_b[:50],
        "common_lines": common,
    }


@router.get("/")
def root():
    return {"status": "running", "service": "ai-compare", "version": "1.0.0"}


@router.post("/compare")
async def compare_documents(
    file_a: Optional[UploadFile] = File(None),
    file_b: Optional[UploadFile] = File(None),
    document_id_a: Optional[str] = Body(None),
    document_id_b: Optional[str] = Body(None),
    mode: str = Body("semantic"),
    llm_provider: Optional[str] = Body("anthropic"),
):
    """
    Compare two documents — each side accepts either a `file` or a `document_id`.
    mode='text'     → line-level diff (no LLM)
    mode='semantic' → AI-powered semantic difference analysis
    """
    content_a, filename_a, _ = await resolve_upload(file_a, document_id_a)
    content_b, filename_b, _ = await resolve_upload(file_b, document_id_b)

    try:
        text_a = extract_text(content_a, filename_a)
        text_b = extract_text(content_b, filename_b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if mode == "text":
        diff = _text_diff(text_a, text_b)
        return {
            "mode": "text",
            "file_a": filename_a,
            "file_b": filename_b,
            "diff": diff,
        }

    # Semantic mode — use LLM
    system = (
        "You are an expert legal and document analyst. Compare two documents and identify "
        "meaningful differences. Focus on: changed clauses, added/removed sections, numerical changes, "
        "changed parties or terms. Ignore formatting differences. Output valid JSON only."
    )
    user = (
        f"Compare these two documents and return a JSON object with:\n"
        f'  "summary": "<2-3 sentence overall comparison>",\n'
        f'  "differences": [<list of specific differences, each as a string>],\n'
        f'  "additions_in_b": [<content present in B but not A>],\n'
        f'  "removals_from_a": [<content present in A but not B>],\n'
        f'  "similarity_score": <0-100 integer>\n\n'
        f"Return ONLY the JSON object — no markdown, no explanation.\n\n"
        f"=== DOCUMENT A ({filename_a}) ===\n{text_a[:6000]}\n\n"
        f"=== DOCUMENT B ({filename_b}) ===\n{text_b[:6000]}"
    )

    try:
        raw = llm.call(system, user, provider=llm_provider or "anthropic")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    return {
        "mode": "semantic",
        "file_a": filename_a,
        "file_b": filename_b,
        "result": result,
        "provider": llm_provider,
    }


@router.post("/compare-versions")
async def compare_versions(
    original: Optional[UploadFile] = File(None),
    translated: Optional[UploadFile] = File(None),
    document_id_original: Optional[str] = Body(None),
    document_id_translated: Optional[str] = Body(None),
    source_language: str = Body("en"),
    target_language: str = Body("hi"),
):
    """Compare original and translated document — check translation quality. Each side accepts either a `file` or a `document_id`."""
    content_orig, filename_orig, _ = await resolve_upload(original, document_id_original)
    content_trans, filename_trans, _ = await resolve_upload(translated, document_id_translated)

    try:
        text_orig = extract_text(content_orig, filename_orig)
        text_trans = extract_text(content_trans, filename_trans)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    src_lang = LANGUAGE_NAMES.get(source_language, source_language)
    tgt_lang = LANGUAGE_NAMES.get(target_language, target_language)

    system = (
        "You are an expert bilingual document reviewer. Evaluate translation quality "
        "and identify any missing content, mistranslations, or omissions. Output valid JSON only."
    )
    user = (
        f"Review the {tgt_lang} translation of this {src_lang} document.\n\n"
        f"Return a JSON object with:\n"
        f'  "quality_score": <0-100>,\n'
        f'  "missing_content": [<list of sections/clauses from original not present in translation>],\n'
        f'  "mistranslations": [<list of suspected mistranslations>],\n'
        f'  "overall_assessment": "<brief assessment>"\n\n'
        f"Return ONLY the JSON object — no markdown, no explanation.\n\n"
        f"=== ORIGINAL ({src_lang}) ===\n{text_orig[:6000]}\n\n"
        f"=== TRANSLATION ({tgt_lang}) ===\n{text_trans[:6000]}"
    )

    try:
        raw = llm.call(system, user, provider="anthropic")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    return {
        "original": filename_orig,
        "translated": filename_trans,
        "source_language": source_language,
        "target_language": target_language,
        "result": result,
    }
