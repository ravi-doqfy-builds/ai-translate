import asyncio
import logging
from fastapi import APIRouter, FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import json
import tempfile
import base64
import io
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from gemini_translator import GeminiTranslator
from sarvam_translator import SarvamTranslator
from openai_translator import OpenAITranslator
from anthropic_translator import AnthropicTranslator
from gemini_reviewer import GeminiReviewer
from openai_reviewer import OpenAIReviewer
from html_to_docx import html_to_docx
from html_to_pdf import html_to_pdf
from docx_converter import convert_docx_stream_to_html
from google.cloud import translate_v2 as translate
from google.api_core.client_options import ClientOptions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Versioned API Router ---
api_v1 = APIRouter(prefix="/api/v1")


# --- Chat / Interact Models ---
class ChatMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system: Optional[str] = None   # optional system prompt override
    provider: Optional[str] = None  # pin to a specific provider; None = auto-fallback


class ChatResponse(BaseModel):
    reply: str
    provider: str      # which provider actually answered


# --- Chat helpers (Gemini → OpenAI → Claude fallback) ---

async def _chat_gemini(messages: List[ChatMessage], system: Optional[str]) -> str:
    """Send a multi-turn chat via Gemini and return the reply text."""
    from google import genai as google_genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    client = google_genai.Client(api_key=api_key, http_options={"timeout": 120})

    config: dict = {}
    if system:
        config["system_instruction"] = system

    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=config if config else None,
    )

    reply = ""
    # Replay history up to (but not including) the last user message so the
    # SDK maintains context, then send the final message.
    for msg in messages[:-1]:
        if msg.role == "user":
            chat.send_message(msg.content)
        # assistant turns are already tracked in chat history automatically

    if messages:
        response = chat.send_message(messages[-1].content)
        reply = response.text.strip()

    return reply


async def _chat_openai(messages: List[ChatMessage], system: Optional[str]) -> str:
    """Send a multi-turn chat via OpenAI and return the reply text."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key)

    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend({"role": m.role, "content": m.content} for m in messages)

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=oai_messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


async def _chat_anthropic(messages: List[ChatMessage], system: Optional[str]) -> str:
    """Send a multi-turn chat via Anthropic Claude and return the reply text."""
    import anthropic as _anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = _anthropic.AsyncAnthropic(api_key=api_key)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    anthropic_messages = [{"role": m.role, "content": m.content} for m in messages]

    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": anthropic_messages,
    }
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)
    return response.content[0].text.strip()


# Provider map — ordered: Gemini first, then OpenAI, then Claude
_CHAT_PROVIDERS: dict = {
    "gemini": _chat_gemini,
    "openai": _chat_openai,
    "anthropic": _chat_anthropic,
}
_CHAT_FALLBACK_ORDER: List[str] = ["gemini", "openai", "anthropic"]


@api_v1.post("/interact/chat", response_model=ChatResponse)
async def interact_chat(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn chat endpoint.

    - Default provider order: Gemini → OpenAI → Claude (first available key wins).
    - Pass `provider` in the request body to pin to a specific one.
    - On failure the next provider in the fallback chain is tried automatically.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Determine which providers to try
    if request.provider:
        provider_key = request.provider.lower()
        if provider_key not in _CHAT_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider '{request.provider}'. Valid: {list(_CHAT_PROVIDERS)}"
            )
        order = [provider_key]
    else:
        order = _CHAT_FALLBACK_ORDER

    last_error: Exception = RuntimeError("No providers available")

    for provider_name in order:
        handler = _CHAT_PROVIDERS[provider_name]
        try:
            reply = await handler(request.messages, request.system)
            logger.info("[Chat] Responded via %s", provider_name)
            return ChatResponse(reply=reply, provider=provider_name)
        except EnvironmentError as exc:
            # API key not configured — silently skip to next provider
            logger.debug("[Chat] %s skipped (no key): %s", provider_name, exc)
            last_error = exc
        except Exception as exc:
            logger.warning("[Chat] %s failed: %s", provider_name, exc)
            last_error = exc

    raise HTTPException(
        status_code=503,
        detail=f"All chat providers failed or are unconfigured. Last error: {last_error}",
    )


app.include_router(api_v1)

# --- Google Translate Client ---
translate_client = None
try:
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        client_options = ClientOptions(quota_project_id=project_id)
        translate_client = translate.Client(client_options=client_options)
        print("DEBUG: Google Translate Client initialized")
    else:
        translate_client = translate.Client()
except Exception as e:
    print(f"WARNING: Google Translate not initialized: {e}")

# --- Provider Registry ---
translator_registry = {}
reviewer_registry = {}

# Initialize Translators
try:
    if os.getenv("GEMINI_API_KEY"):
        translator_registry["gemini"] = GeminiTranslator()
        print("DEBUG: Gemini Translator registered")
except Exception as e:
    print(f"WARNING: Gemini Translator failed: {e}")

try:
    if os.getenv("SARVAM_API_KEY"):
        translator_registry["sarvam"] = SarvamTranslator()
        print("DEBUG: Sarvam Translator registered")
except Exception as e:
    print(f"WARNING: Sarvam Translator failed: {e}")

try:
    if os.getenv("OPENAI_API_KEY"):
        translator_registry["openai"] = OpenAITranslator()
        print("DEBUG: OpenAI Translator registered")
except Exception as e:
    print(f"WARNING: OpenAI Translator failed: {e}")

try:
    if os.getenv("ANTHROPIC_API_KEY"):
        translator_registry["anthropic"] = AnthropicTranslator()
        print("DEBUG: Anthropic Translator registered")
except Exception as e:
    print(f"WARNING: Anthropic Translator failed: {e}")

# Initialize Reviewers
try:
    if os.getenv("GEMINI_API_KEY"):
        reviewer_registry["gemini"] = GeminiReviewer()
        print("DEBUG: Gemini Reviewer registered")
except Exception as e:
    print(f"WARNING: Gemini Reviewer failed: {e}")

try:
    if os.getenv("OPENAI_API_KEY"):
        reviewer_registry["openai"] = OpenAIReviewer()
        print("DEBUG: OpenAI Reviewer registered")
except Exception as e:
    print(f"WARNING: OpenAI Reviewer failed: {e}")


# --- Data Models ---
class EvaluationRequest(BaseModel):
    original_text: str
    translated_text: str
    target_language: str
    reviewer_provider: str = "gemini"


class TranslateURLRequest(BaseModel):
    url: str
    target_language: str = "hi"
    # None or "google" → Google Translate; any other value (e.g. "gemini") → LLM
    llm_provider: Optional[str] = None


class TranslateBase64Request(BaseModel):
    file_data: str    # base64-encoded file bytes
    filename: str     # used to detect file type (.pdf or .docx)
    target_language: str = "hi"
    # None or "google" → Google Translate; any other value → LLM
    llm_provider: Optional[str] = None


# --- Helper Functions ---
def extract_pdf_pages_html(file_bytes: bytes) -> list:
    """Extract each PDF page as its own HTML string using PyMuPDF."""
    import pymupdf
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        page_html = page.get_text("html")
        pages.append(
            f'<div class="page" style="margin-bottom:2em;padding:1em;border-bottom:1px solid #eee;">'
            f'{page_html}</div>'
        )
    doc.close()
    return pages


def extract_pdf_to_html(file_bytes: bytes) -> str:
    """Extract all PDF pages and combine into a single HTML document."""
    pages = extract_pdf_pages_html(file_bytes)
    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>'
        'body{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;}'
        '.page{page-break-after:always;}'
        '</style></head><body>'
        + ''.join(pages)
        + '</body></html>'
    )


def _google_translate_html(html: str, target_language: str) -> str:
    """Run Google Translate on an HTML string, handling style/image placeholders."""
    import re

    if not translate_client:
        raise HTTPException(status_code=503, detail="Google Translate not configured")

    style_placeholders = {}

    def replace_style(match):
        placeholder = f"STYLE_PLACEHOLDER_{len(style_placeholders)}"
        style_placeholders[placeholder] = match.group(0)
        return f'<div class="{placeholder}"></div>'

    html_for_translation = re.sub(r'<style[^>]*>[\s\S]*?</style>', replace_style, html)

    image_placeholders = {}

    def replace_img(match):
        quote = match.group(1)
        content_val = match.group(2)
        placeholder = f"IMG_PLACEHOLDER_{len(image_placeholders)}"
        image_placeholders[placeholder] = content_val
        return f'src={quote}{placeholder}{quote}'

    html_for_translation = re.sub(r'src=(["\'])(data:image/[^"\']+)\1', replace_img, html_for_translation)

    result = translate_client.translate(html_for_translation, target_language=target_language, format_='html')
    translated_html = result['translatedText']
    translated_html = translated_html.replace('&#39;', "'").replace('&quot;', '"').replace('&amp;', '&')

    for placeholder, style_block in style_placeholders.items():
        translated_html = re.sub(rf'<div class="{placeholder}"\s*>\s*</div>', style_block, translated_html)
        if placeholder in translated_html:
            translated_html = translated_html.replace(placeholder, "")
            translated_html = style_block + translated_html

    for placeholder, img_content in image_placeholders.items():
        translated_html = translated_html.replace(placeholder, img_content)

    return translated_html


# Terms that must NEVER be translated (shared between translator and reviewer)
_REVIEW_PROTECTED_TERMS = [
    "L&T Finance Holdings Limited", "L&T Finance Holdings",
    "L&T Finance Limited", "L&T Housing Finance Limited",
    "L&T Housing Finance", "L&T Finance", "L&T",
    "Pvt. Ltd.", "Pvt Ltd", "Ltd.", "Co.", "Inc.", "Corp.", "Limited", "M/s",
    "NACH", "CIBIL", "NEFT", "RTGS", "MSME", "NBFC", "FOIR",
    "RBI", "SEBI", "GST", "PAN", "TDS", "EMI", "UPI", "KYC",
    "NPA", "MOU", "LOA", "NOC", "CIN", "DIN", "LLPIN", "SRN",
    "ROI", "IRR", "APR",
]


def _protect_for_review(html: str):
    """Replace protected terms and [[...]] tokens with placeholders before LLM review."""
    import re
    term_map = {}
    text = html
    for i, term in enumerate(_REVIEW_PROTECTED_TERMS):
        if term in text:
            token = f"__RTERM_{i}__"
            term_map[token] = term
            text = text.replace(term, token)
    bracket_tokens = re.findall(r'\[\[.*?\]\]', text)
    for j, bt in enumerate(bracket_tokens):
        token = f"__RBRACKET_{j}__"
        if token not in term_map:
            term_map[token] = bt
            text = text.replace(bt, token, 1)
    return text, term_map


def _restore_after_review(html: str, term_map: dict) -> str:
    for token, original in term_map.items():
        html = html.replace(token, original)
    return html


async def _auto_review_translation(html: str, target_language: str) -> str:
    """
    Post-translation pass: use OpenAI (fallback: Anthropic) to find and fix
    any visible text that is still in English and should be translated.
    """
    language_names = {
        'hi': 'Hindi', 'te': 'Telugu', 'mr': 'Marathi',
        'bn': 'Bengali', 'kn': 'Kannada', 'ta': 'Tamil',
        'gu': 'Gujarati', 'or': 'Odia', 'ml': 'Malayalam',
        'pa': 'Punjabi', 'as': 'Assamese', 'rajasthani': 'Hindi'
    }
    target_lang_name = language_names.get(target_language, target_language)

    # Protect known terms so the reviewer cannot accidentally translate them
    protected_html, term_map = _protect_for_review(html)

    token_note = ""
    if term_map:
        token_note = (
            "\n⚠️  TOKEN PRESERVATION: The HTML contains __RTERM_N__ and __RBRACKET_N__ tokens "
            "(e.g., __RTERM_0__, __RBRACKET_2__). These are placeholders for proper nouns and IDs. "
            "You MUST copy them into the output EXACTLY as-is — never translate, modify, or remove them.\n"
        )

    system_msg = (
        "You are a translation quality reviewer for corporate and regulatory documents. "
        "Your job is to find regular English text that was missed during translation and translate it. "
        "CRITICAL: You must NEVER translate company names (L&T, L&T Finance, M/s, Pvt Ltd, Ltd., Limited), "
        "regulatory abbreviations (RBI, SEBI, GST, PAN, TDS, EMI, NACH, CIBIL, KYC, NEFT, RTGS, UPI, "
        "MSME, NBFC), or placeholder tokens (__RTERM_N__, __RBRACKET_N__). These are INTENTIONALLY kept "
        "in Roman/English script and must remain unchanged."
    )

    prompt = f"""You are a translation quality reviewer for corporate and regulatory documents.
{token_note}
The following HTML document was translated to {target_lang_name}, but some regular text may still be in English.

YOUR TASK:
1. Find any visible regular text that is still in English and should be in {target_lang_name}.
2. Translate ONLY those missed English text segments to {target_lang_name}.
3. Return the complete corrected HTML.

ABSOLUTE DO-NOT-CHANGE LIST — these MUST stay exactly as-is (they are intentionally in Roman script):
- HTML tags, attributes, class names, inline styles, <style> blocks
- Placeholder tokens: __RTERM_N__, __RBRACKET_N__ (e.g., __RTERM_0__, __RBRACKET_2__)
- Numbers, dates, currency amounts (e.g., ₹1,00,000, 12/03/2024, 18%)

NOTE: The following are PROPER NOUNS that are CORRECT to appear in Roman/English script in a {target_lang_name} document. Do NOT translate them:
- Company names: L&T, L&T Finance, L&T Finance Limited, L&T Finance Holdings, L&T Housing Finance, M/s, Pvt Ltd, Ltd., Limited, Co., Inc., Corp.
- Regulatory terms: RBI, SEBI, GST, PAN, TDS, EMI, NACH, CIBIL, KYC, NEFT, RTGS, UPI, MSME, NPA, NBFC, MOU, LOA, NOC, CIN, DIN, LLPIN, SRN, ROI, IRR, APR, FOIR

Return ONLY the corrected HTML. No explanations, no markdown code blocks.

HTML to review:
{protected_html}

Corrected HTML:"""

    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            reviewed = response.choices[0].message.content.strip()
            if reviewed.startswith("```"):
                lines = reviewed.split('\n')
                if len(lines) > 2:
                    reviewed = '\n'.join(lines[1:-1])
            reviewed = _restore_after_review(reviewed, term_map)
            print("[AutoReview] OpenAI review pass completed")
            return reviewed
        except Exception as e:
            print(f"[AutoReview] OpenAI failed, trying Anthropic: {e}")

    # Fallback: Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.AsyncAnthropic(api_key=anthropic_key)
            response = await client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                max_tokens=8192,
                system=system_msg,
                messages=[{"role": "user", "content": prompt}],
            )
            reviewed = response.content[0].text.strip()
            if reviewed.startswith("```"):
                lines = reviewed.split('\n')
                if len(lines) > 2:
                    reviewed = '\n'.join(lines[1:-1])
            reviewed = _restore_after_review(reviewed, term_map)
            print("[AutoReview] Anthropic review pass completed")
            return reviewed
        except Exception as e:
            print(f"[AutoReview] Anthropic failed: {e}")

    print("[AutoReview] No reviewer available, skipping")
    return html


async def _do_translate_docx_google(content: bytes, target_language: str) -> dict:
    """Core DOCX → Google Translate processing (works on raw bytes)."""
    fileobj = io.BytesIO(content)
    html_output = convert_docx_stream_to_html(fileobj)

    if target_language.lower() == 'rajasthani':
        target_language = 'hi'

    translated_html = _google_translate_html(html_output, target_language)

    translated_docx_b64 = None
    translated_pdf_b64 = None
    try:
        docx_io = html_to_docx(translated_html)
        translated_docx_b64 = base64.b64encode(docx_io.getvalue()).decode('utf-8')
        pdf_io = html_to_pdf(translated_html)
        translated_pdf_b64 = base64.b64encode(pdf_io.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Failed to generate DOCX/PDF: {e}")

    return {
        "html": translated_html,
        "original_html": html_output,
        "language": target_language,
        "translated_docx_b64": translated_docx_b64,
        "translated_pdf_b64": translated_pdf_b64
    }


async def _do_translate_docx_llm(content: bytes, target_language: str, llm_provider: str) -> dict:
    """Core DOCX → LLM translation processing (works on raw bytes)."""
    import time as _time

    target_lang_code = 'hi' if target_language.lower() == 'rajasthani' else target_language

    fd_in, input_path = tempfile.mkstemp(suffix=".docx")
    fd_out, output_path = tempfile.mkstemp(suffix=".docx")
    os.close(fd_in)
    os.close(fd_out)

    try:
        with open(input_path, "wb") as f:
            f.write(content)

        original_fileobj = io.BytesIO(content)
        original_html = convert_docx_stream_to_html(original_fileobj)

        provider = translator_registry.get(llm_provider)
        if not provider:
            raise HTTPException(status_code=503, detail=f"Translator '{llm_provider}' not configured")

        _t_start = _time.time()
        print(f"[DOCX-LLM] Starting translation via {llm_provider} → {target_lang_code}")
        provider.translate_docx(input_path, target_lang_code, output_path)
        print(f"[DOCX-LLM] Done in {_time.time()-_t_start:.1f}s")

        with open(output_path, "rb") as f:
            translated_content = f.read()
            translated_docx_b64 = base64.b64encode(translated_content).decode('utf-8')
            translated_fileobj = io.BytesIO(translated_content)
            translated_html = convert_docx_stream_to_html(translated_fileobj)

        # Auto-review: fix any missed English text
        translated_html = await _auto_review_translation(translated_html, target_language)

        translated_pdf_b64 = None
        try:
            pdf_io = html_to_pdf(translated_html)
            translated_pdf_b64 = base64.b64encode(pdf_io.getvalue()).decode('utf-8')
        except Exception as pdf_err:
            print(f"Warning: PDF generation failed: {pdf_err}")

        return {
            "html": translated_html,
            "original_html": original_html,
            "language": target_language,
            "provider": llm_provider,
            "translated_docx_b64": translated_docx_b64,
            "translated_pdf_b64": translated_pdf_b64
        }
    finally:
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)


def _do_translate_pdf_google(content: bytes, target_language: str) -> dict:
    """Core PDF → Google Translate processing (works on raw bytes)."""
    if target_language.lower() == 'rajasthani':
        target_language = 'hi'

    original_html = extract_pdf_to_html(content)
    translated_html = _google_translate_html(original_html, target_language)

    translated_pdf_b64 = None
    try:
        pdf_io = html_to_pdf(translated_html)
        translated_pdf_b64 = base64.b64encode(pdf_io.getvalue()).decode('utf-8')
    except Exception as pdf_err:
        print(f"Warning: PDF generation failed: {pdf_err}")

    return {
        "html": translated_html,
        "original_html": original_html,
        "language": target_language,
        "translated_docx_b64": None,
        "translated_pdf_b64": translated_pdf_b64
    }


PAGES_PER_CHUNK = 3  # keep each LLM call well under token limits

_HTML_HEADER = (
    '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>'
    'body{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;}'
    '.page{page-break-after:always;}'
    '</style></head><body>'
)


async def _do_translate_pdf_llm(content: bytes, target_language: str, llm_provider: str) -> dict:
    """Translate PDF via LLM using sequential 3-page chunks to stay under token limits."""
    import re as _re

    target_lang_code = 'hi' if target_language.lower() == 'rajasthani' else target_language

    pages = extract_pdf_pages_html(content)
    original_html = extract_pdf_to_html(content)

    provider = translator_registry.get(llm_provider)
    if not provider:
        raise HTTPException(status_code=503, detail=f"Translator '{llm_provider}' not configured")

    chunks = [pages[i:i + PAGES_PER_CHUNK] for i in range(0, len(pages), PAGES_PER_CHUNK)]
    translated_pages = []

    for idx, chunk_pages in enumerate(chunks):
        chunk_html = _HTML_HEADER + ''.join(chunk_pages) + '</body></html>'
        print(f"[PDF-LLM] Chunk {idx + 1}/{len(chunks)} ({len(chunk_pages)} pages) via {llm_provider}")
        try:
            result = await asyncio.to_thread(provider.translate_html, chunk_html, target_lang_code)
            body_match = _re.search(r'<body[^>]*>([\s\S]*?)</body>', result["translated_html"], _re.IGNORECASE)
            translated_pages.append(body_match.group(1) if body_match else result["translated_html"])
        except Exception as e:
            print(f"[PDF-LLM] Chunk {idx + 1} failed ({e}), keeping original")
            translated_pages.extend(chunk_pages)

    translated_html = _HTML_HEADER + ''.join(translated_pages) + '</body></html>'

    # Auto-review: fix any missed English text
    translated_html = await _auto_review_translation(translated_html, target_lang_code)

    translated_pdf_b64 = None
    try:
        pdf_io = html_to_pdf(translated_html)
        translated_pdf_b64 = base64.b64encode(pdf_io.getvalue()).decode('utf-8')
    except Exception as pdf_err:
        print(f"Warning: PDF generation failed: {pdf_err}")

    return {
        "html": translated_html,
        "original_html": original_html,
        "language": target_language,
        "provider": llm_provider,
        "translated_docx_b64": None,
        "translated_pdf_b64": translated_pdf_b64
    }


# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "running", "service": "ai-translate", "version": "1.0.0"}


@app.post("/translate-docx")
async def translate_docx_google(
    file: UploadFile = File(...),
    target_language: str = Body("hi")
):
    """Translate DOCX using Google Translate (HTML mode)"""
    if not file.filename.lower().endswith('.docx'):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    try:
        content = await file.read()
        return await _do_translate_docx_google(content, target_language)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate-docx-llm")
async def translate_docx_llm(
    file: UploadFile = File(...),
    target_language: str = Body("hi"),
    llm_provider: str = Body("gemini")
):
    """Translate DOCX natively using an LLM provider (preserves layout, images, tables)"""
    if not file.filename.lower().endswith('.docx'):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    try:
        content = await file.read()
        return await _do_translate_docx_llm(content, target_language, llm_provider)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate-pdf-llm")
async def translate_pdf_llm(
    file: UploadFile = File(...),
    target_language: str = Body("hi"),
    llm_provider: str = Body("gemini")
):
    """Translate PDF using a single LLM call"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    try:
        content = await file.read()
        return await _do_translate_pdf_llm(content, target_language, llm_provider)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF Translation failed: {str(e)}")


@app.post("/translate-pdf-llm/stream")
async def translate_pdf_llm_stream(
    file: UploadFile = File(...),
    target_language: str = Body("hi"),
    llm_provider: str = Body("gemini")
):
    """SSE endpoint — sequential 3-page chunks, emits a chunk event after each, then done."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    content = await file.read()
    target_lang_code = 'hi' if target_language.lower() == 'rajasthani' else target_language

    pages = extract_pdf_pages_html(content)
    original_html = extract_pdf_to_html(content)

    provider = translator_registry.get(llm_provider)
    if not provider:
        raise HTTPException(status_code=503, detail=f"Translator '{llm_provider}' not configured")

    chunks = [pages[i:i + PAGES_PER_CHUNK] for i in range(0, len(pages), PAGES_PER_CHUNK)]

    import re as _re

    async def event_gen():
        translated_pages = []

        for idx, chunk_pages in enumerate(chunks):
            if idx > 0:
                await asyncio.sleep(2)  # breathing room between chunks to avoid rate limits

            chunk_html = _HTML_HEADER + ''.join(chunk_pages) + '</body></html>'
            print(f"[PDF-Stream] Chunk {idx + 1}/{len(chunks)} ({len(chunk_pages)} pages)")

            # Run translation in a thread; send SSE keep-alive pings every 5s
            # so the browser never drops the connection during long API calls / retries
            task = asyncio.ensure_future(
                asyncio.to_thread(provider.translate_html, chunk_html, target_lang_code)
            )
            while not task.done():
                yield ": keep-alive\n\n"
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except asyncio.TimeoutError:
                    pass  # still running — ping again

            try:
                result = task.result()
                body_match = _re.search(r'<body[^>]*>([\s\S]*?)</body>', result["translated_html"], _re.IGNORECASE)
                chunk_body = body_match.group(1) if body_match else result["translated_html"]
            except Exception as e:
                print(f"[PDF-Stream] Chunk {idx + 1} failed ({e}), keeping original")
                chunk_body = ''.join(chunk_pages)

            translated_pages.append(chunk_body)
            payload = json.dumps({
                "type": "chunk",
                "index": idx,
                "total": len(chunks),
                "done": idx + 1,
                "html": chunk_body,
            })
            yield f"data: {payload}\n\n"

        full_html = _HTML_HEADER + ''.join(translated_pages) + '</body></html>'

        translated_pdf_b64 = None
        try:
            pdf_io = html_to_pdf(full_html)
            translated_pdf_b64 = base64.b64encode(pdf_io.getvalue()).decode('utf-8')
        except Exception as pdf_err:
            print(f"Warning: PDF generation failed: {pdf_err}")

        done_payload = json.dumps({
            "type": "done",
            "html": full_html,
            "original_html": original_html,
            "language": target_language,
            "provider": llm_provider,
            "translated_pdf_b64": translated_pdf_b64,
        })
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/translate-pdf")
async def translate_pdf_google(
    file: UploadFile = File(...),
    target_language: str = Body("hi")
):
    """Translate PDF using Google Translate"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    try:
        content = await file.read()
        return _do_translate_pdf_google(content, target_language)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF Translation failed: {str(e)}")


@app.post("/translate-url")
async def translate_from_url(request: TranslateURLRequest):
    """
    Download a document from a URL and translate it.

    - Supports .pdf and .docx files (detected from URL path or Content-Type header).
    - llm_provider=null / "google"  → Google Translate
    - llm_provider="gemini" / "openai" / etc. → LLM-based translation
    """
    import requests as req_lib

    try:
        resp = await asyncio.to_thread(
            lambda: req_lib.get(request.url, timeout=60, allow_redirects=True)
        )
        resp.raise_for_status()
        content = resp.content

        # Detect file type from URL path first, then Content-Type
        url_path = request.url.lower().split('?')[0]
        content_type = resp.headers.get('Content-Type', '').lower()

        if url_path.endswith('.pdf') or 'pdf' in content_type:
            file_type = 'pdf'
        elif url_path.endswith('.docx') or 'officedocument.wordprocessingml' in content_type:
            file_type = 'docx'
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot detect file type. URL must point to a .pdf or .docx file."
            )

        use_llm = bool(request.llm_provider and request.llm_provider.lower() != 'google')

        if file_type == 'pdf':
            if use_llm:
                return await _do_translate_pdf_llm(content, request.target_language, request.llm_provider)
            else:
                return _do_translate_pdf_google(content, request.target_language)
        else:
            if use_llm:
                return await _do_translate_docx_llm(content, request.target_language, request.llm_provider)
            else:
                return await _do_translate_docx_google(content, request.target_language)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate-base64")
async def translate_from_base64(request: TranslateBase64Request):
    """
    Translate a base64-encoded document file.

    - filename is used to detect the file type (.pdf or .docx).
    - llm_provider=null / "google"  → Google Translate
    - llm_provider="gemini" / "openai" / etc. → LLM-based translation
    """
    try:
        try:
            content = base64.b64decode(request.file_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 data")

        filename_lower = request.filename.lower()
        if filename_lower.endswith('.pdf'):
            file_type = 'pdf'
        elif filename_lower.endswith('.docx'):
            file_type = 'docx'
        else:
            raise HTTPException(
                status_code=400,
                detail="filename must end with .pdf or .docx"
            )

        use_llm = bool(request.llm_provider and request.llm_provider.lower() != 'google')

        if file_type == 'pdf':
            if use_llm:
                return await _do_translate_pdf_llm(content, request.target_language, request.llm_provider)
            else:
                return _do_translate_pdf_google(content, request.target_language)
        else:
            if use_llm:
                return await _do_translate_docx_llm(content, request.target_language, request.llm_provider)
            else:
                return await _do_translate_docx_google(content, request.target_language)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/evaluate-translation")
async def evaluate_translation(request: EvaluationRequest):
    """Evaluate translation quality using a reviewer model"""
    try:
        reviewer = reviewer_registry.get(request.reviewer_provider)
        if not reviewer:
            raise HTTPException(status_code=503, detail=f"Reviewer '{request.reviewer_provider}' not configured")

        evaluation = reviewer.evaluate_translation(
            request.original_text,
            request.translated_text,
            request.target_language
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-correction")
async def save_correction(
    original_text: str = Body(...),
    translated_text: str = Body(...),
    corrected_text: str = Body(...),
    target_language: str = Body(...),
    provider: str = Body(...)
):
    """Save human corrections for future fine-tuning"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, "corrections.jsonl")

        entry = {
            "timestamp": uuid.uuid4().hex,
            "original_text": original_text,
            "llm_output": translated_text,
            "human_correction": corrected_text,
            "language": target_language,
            "provider": provider
        }

        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return {"status": "success", "message": "Correction saved for future training"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download-docx")
async def download_docx(
    html_content: str = Body(..., embed=True),
    filename: str = Body("translated_document.docx", embed=True)
):
    """Convert HTML to DOCX and stream as download"""
    try:
        docx_io = html_to_docx(html_content)
        return StreamingResponse(
            docx_io,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX conversion failed: {str(e)}")


@app.post("/download-pdf")
async def download_pdf(
    html_content: str = Body(..., embed=True),
    filename: str = Body("translated_document.pdf", embed=True)
):
    """Convert HTML to PDF and stream as download"""
    try:
        pdf_io = html_to_pdf(html_content)
        return StreamingResponse(
            pdf_io,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")
