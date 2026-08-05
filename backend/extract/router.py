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
print(f"[extract] service ready | flow={llm.flow}" + (f" | model={llm.model_id}" if llm.flow == "server" else ""))


@router.get("/")
def root():
    return {"status": "running", "service": "ai-extract", "version": "1.0.0"}


def _parse_json_response(raw: str) -> dict | list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    return json.loads(raw)


@router.post("/extract")
async def extract_fields(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Body(None),
    fields: str = Body(...),
    llm_provider: Optional[str] = Body("anthropic"),
):
    """
    Extract specific named fields from a document — pass either `file` or a `document_id`.
    `fields` is a JSON array of field names, e.g. '["borrower_name", "loan_amount"]'.
    Returns a JSON object with extracted values.
    """
    content, filename, _ = await resolve_upload(file, document_id)
    try:
        text = extract_text(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        field_list = json.loads(fields)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`fields` must be a valid JSON array of strings.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

    fields_str = "\n".join(f"- {f}" for f in field_list)
    system = (
        "You are an expert at structured data extraction from legal and corporate documents. "
        "Extract only the fields requested. Return null for fields not found. Output valid JSON only."
    )
    user = (
        f"Extract the following fields from the document and return them as a JSON object.\n"
        f"Fields to extract:\n{fields_str}\n\n"
        f"Rules:\n"
        f"- Use the exact field names as keys.\n"
        f"- Set the value to null if the field is not found.\n"
        f"- Return ONLY a JSON object — no markdown, no explanation.\n\n"
        f"Document:\n{text[:12000]}"
    )

    try:
        raw = llm.call(system, user, provider=llm_provider or "anthropic")
        result = _parse_json_response(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    return {"extracted": result, "provider": llm_provider}


@router.post("/extract-schema")
async def extract_with_schema(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Body(None),
    schema: str = Body(...),
    llm_provider: Optional[str] = Body("anthropic"),
):
    """Extract structured data using a JSON Schema definition. Pass either `file` or a `document_id`."""
    content, filename, _ = await resolve_upload(file, document_id)
    try:
        text = extract_text(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        schema_obj = json.loads(schema)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`schema` must be valid JSON.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

    system = (
        "You are an expert at structured data extraction. Extract data from documents following "
        "the provided JSON Schema exactly. Output valid JSON only."
    )
    user = (
        f"Extract data from the document following this JSON Schema:\n"
        f"{json.dumps(schema_obj, indent=2)}\n\n"
        f"Return ONLY a JSON object conforming to the schema — no markdown, no explanation.\n\n"
        f"Document:\n{text[:12000]}"
    )

    try:
        raw = llm.call(system, user, provider=llm_provider or "anthropic")
        result = _parse_json_response(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    return {"extracted": result, "provider": llm_provider}


@router.post("/extract-table")
async def extract_tables(
    file: Optional[UploadFile] = File(None),
    document_id: Optional[str] = Body(None),
    llm_provider: Optional[str] = Body("anthropic"),
):
    """Extract all tables from the document as structured JSON. Pass either `file` or a `document_id`."""
    content, filename, _ = await resolve_upload(file, document_id)
    try:
        text = extract_text(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the document.")

    system = (
        "You are an expert at extracting tables from documents. "
        "Identify all tables and return them as structured JSON. Output valid JSON only."
    )
    user = (
        f"Identify and extract ALL tables from the document.\n"
        f"Return a JSON array where each element represents a table:\n"
        f'  {{"title": "<table title or null>", "headers": ["col1", "col2", ...], "rows": [["val1", "val2", ...], ...]}}\n\n'
        f"Return ONLY the JSON array — no markdown, no explanation.\n\n"
        f"Document:\n{text[:12000]}"
    )

    try:
        raw = llm.call(system, user, provider=llm_provider or "anthropic")
        tables = _parse_json_response(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Table extraction failed: {e}")

    return {"tables": tables, "provider": llm_provider}
