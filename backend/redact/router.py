"""
Redact feature router.

Endpoints:
  POST /redact-by-role   — AI identifies what to redact based on the viewer's role
  POST /redact-custom    — Redact by highlight color or by page/paragraph/line position
"""

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from llm_client import LLMClient
from doc_reader import extract_text
from document_store import resolve_upload
from redact_engine import (
    redact_docx_by_highlight_color,
    redact_docx_by_phrases,
    redact_docx_by_positions,
    redact_pdf_by_highlight_color,
    redact_pdf_by_phrases,
    redact_pdf_by_positions,
)

router = APIRouter()

# ── Service initialisation ─────────────────────────────────────────────────────
_FLOW = os.getenv("LLM_FLOW", os.getenv("TRANSLATION_FLOW", "direct"))

llm = LLMClient()  # always available as fallback

if _FLOW == "server":
    from bedrock_redact_service import BedrockRedactService
    _bedrock_svc = BedrockRedactService()
    print(f"[redact] BEDROCK registered | model={_bedrock_svc.model_id}")
else:
    _bedrock_svc = None
    print(f"[redact] service ready | flow=direct")

ROLES = {
    "legal_ops":  "Legal Operations",
    "admin":      "System Administrator",
    "hr":         "Human Resources",
    "finance":    "Finance",
    "executive":  "Executive / C-Suite",
    "external":   "External Party",
}

ROLE_INSTRUCTIONS = {
    "legal_ops": (
        "The viewer is in Legal Operations. Redact: internal counsel notes, privileged attorney-client "
        "communications, litigation strategy, settlement figures, personal opinions of counsel, "
        "and any text marked 'PRIVILEGED' or 'CONFIDENTIAL — ATTORNEY CLIENT'."
    ),
    "admin": (
        "The viewer is a System Administrator. Redact: personally identifiable information (PII), "
        "passwords, API keys, access credentials, social security numbers, bank account numbers, "
        "credit card numbers, and private employee records."
    ),
    "hr": (
        "The viewer is in Human Resources. Redact: salary and compensation figures, performance review "
        "details, disciplinary records, personal health information, background check results, "
        "and any sensitive personal details beyond basic contact information."
    ),
    "finance": (
        "The viewer is in Finance. Redact: unreleased revenue projections, confidential pricing, "
        "M&A deal terms, loan account numbers, investor names, and non-public financial forecasts."
    ),
    "executive": (
        "The viewer is an Executive / C-Suite member. Redact only the most sensitive operational "
        "details: individual employee PII, raw customer data, and privileged legal opinions."
    ),
    "external": (
        "The viewer is an External Party (outside the organisation). Apply maximum redaction: "
        "redact all internal financial data, employee names and roles, internal project names, "
        "confidential business terms, pricing, strategy, legal matters, and any internal identifiers."
    ),
}


def _is_pdf(content: bytes, filename: str) -> bool:
    return filename.lower().endswith(".pdf") or content[:4] == b"%PDF"


def _stem(filename: str) -> str:
    """Return filename without extension."""
    return filename.rsplit(".", 1)[0] if "." in filename else filename


@router.get("/")
def root():
    return {"status": "running", "service": "ai-redact", "version": "1.0.0"}


@router.post("/redact-by-role")
async def redact_by_role(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Form(None),
    role: str = Form(...),
    llm_provider: Optional[str] = Form("anthropic"),
):
    """AI-powered role-based document redaction. Pass either `file` or a `document_id`."""
    content, filename, mime_type = await resolve_upload(file, document_id)
    fname = filename or "document"
    print(f"[redact] redact-by-role | file={fname} | role={role} | flow={_FLOW} | provider={llm_provider}")

    # ── Bedrock path: vision-capable, sends PDF/image directly ────────────────
    if _bedrock_svc is not None:
        try:
            phrases = await _bedrock_svc.identify_phrases_for_role(content, mime_type, fname, role)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bedrock call failed: {e}")
    else:
        # ── Direct LLM path: text extraction then prompt ───────────────────────
        try:
            text = extract_text(content, fname)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

        role_label = ROLES.get(role, role)
        role_instr = ROLE_INSTRUCTIONS.get(role, f"The viewer's role is '{role_label}'.")

        system = (
            "You are a document security specialist. Identify sensitive information to redact "
            "based on the viewer's role. Return ONLY valid JSON — no markdown, no commentary."
        )
        user = (
            f"{role_instr}\n\n"
            f"From the document below, list every exact phrase, name, number, or passage that "
            f"should be redacted for this viewer.\n\n"
            f"Rules:\n"
            f"1. Each item must be VERBATIM text that appears in the document.\n"
            f"2. Be thorough — short fragments (e.g. a single number) are valid entries.\n"
            f"3. Do not include generic words or structural headings unless they reveal sensitive info.\n\n"
            f"Return JSON: {{\"redact_phrases\": [\"exact phrase 1\", \"exact phrase 2\", ...]}}\n\n"
            f"Document:\n{text[:12000]}"
        )

        try:
            raw = llm.call(system, user, provider=llm_provider or "anthropic")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

        try:
            phrases = json.loads(raw).get("redact_phrases", [])
        except json.JSONDecodeError:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            try:
                phrases = json.loads(match.group()) if match else []
            except Exception:
                phrases = []

    if not phrases:
        raise HTTPException(status_code=422, detail="AI could not identify phrases to redact for this role.")

    print(f"[redact] redact-by-role DONE | phrases={len(phrases)} | role={role} | flow={_FLOW}")

    try:
        if _is_pdf(content, fname):
            redacted = redact_pdf_by_phrases(content, phrases)
            media_type = "application/pdf"
            out_name = f"{_stem(fname)}_redacted.pdf"
        else:
            redacted = redact_docx_by_phrases(content, phrases)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            out_name = f"{_stem(fname)}_redacted.docx"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redaction failed: {e}")

    return StreamingResponse(
        io.BytesIO(redacted),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@router.post("/redact-custom")
async def redact_custom(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Form(None),
    mode: str = Form(...),             # "color" | "position"
    hex_color: Optional[str] = Form(None),
    positions_json: Optional[str] = Form(None),  # JSON string
):
    """Custom document redaction by highlight colour or by position. Pass either `file` or a `document_id`."""
    content, filename, _ = await resolve_upload(file, document_id)
    fname = filename or "document"
    is_pdf = _is_pdf(content, fname)

    print(f"[redact] redact-custom | file={fname} | mode={mode} | is_pdf={is_pdf}")

    try:
        if mode == "color":
            if not hex_color:
                raise HTTPException(status_code=400, detail="hex_color is required for color mode.")

            if is_pdf:
                redacted = redact_pdf_by_highlight_color(content, hex_color)
                media_type = "application/pdf"
                out_name = f"{_stem(fname)}_redacted.pdf"
            else:
                redacted = redact_docx_by_highlight_color(content, hex_color)
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                out_name = f"{_stem(fname)}_redacted.docx"

        elif mode == "position":
            if not positions_json:
                raise HTTPException(status_code=400, detail="positions_json is required for position mode.")
            try:
                positions = json.loads(positions_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="positions_json is not valid JSON.")

            if is_pdf:
                redacted = redact_pdf_by_positions(content, positions)
                media_type = "application/pdf"
                out_name = f"{_stem(fname)}_redacted.pdf"
            else:
                redacted = redact_docx_by_positions(content, positions)
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                out_name = f"{_stem(fname)}_redacted.docx"

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Use 'color' or 'position'.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redaction failed: {e}")

    print(f"[redact] redact-custom DONE | file={fname} | mode={mode}")

    return StreamingResponse(
        io.BytesIO(redacted),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
