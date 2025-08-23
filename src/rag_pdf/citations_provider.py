from __future__ import annotations
"""
Provider integrations for retrieving PDF citations and wiring them to the
highlighter. Currently supports Anthropic Claude Messages API using raw httpx.

Env vars:
- ANTHROPIC_API_KEY (required)
- ANTHROPIC_MODEL (optional, default: "claude-3-5-sonnet-latest")
"""

import base64
import json
import os
from typing import Any, Dict, List, Tuple

import httpx


from .config import Settings


ANTHROPIC_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")


def _read_pdf_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def claude_query_with_pdf_and_citations(
    pdf_path: str,
    question: str,
    settings: Settings,
    title: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call Anthropic Messages API with a PDF attachment and request citations.

    Returns the raw JSON response.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required but not set. See README.md")

    pdf_b64 = _read_pdf_b64(pdf_path)
    headers = {
        "content-type": "application/json",
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model or settings.anthropic_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        **({"title": title} if title else {}),
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
    }
    with httpx.Client(timeout=120) as client:
        r = client.post(ANTHROPIC_URL, headers=headers, json=body)
        r.raise_for_status()
        return r.json()


def extract_page_location_citations_from_claude(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract page_location citations from Claude response JSON.
    Returns a list of dicts with keys cited_text, start_page_number, end_page_number.
    """
    out: List[Dict[str, Any]] = []
    for content in resp_json.get("content", []) or []:
        # content blocks may be objects with optional 'citations'
        citations = content.get("citations") if isinstance(content, dict) else None
        if citations:
            for c in citations:
                if c.get("type") == "page_location":
                    out.append({
                        "cited_text": c.get("cited_text", ""),
                        "start_page_number": c.get("start_page_number"),
                        "end_page_number": c.get("end_page_number", c.get("start_page_number")),
                    })
    # Some SDKs nest citations differently; also walk choices-like structures
    for item in resp_json.get("choices", []) or []:
        msg = (item.get("message") or {}).get("content") or []
        for blk in msg:
            cits = blk.get("citations") if isinstance(blk, dict) else None
            if cits:
                for c in cits:
                    if c.get("type") == "page_location":
                        out.append({
                            "cited_text": c.get("cited_text", ""),
                            "start_page_number": c.get("start_page_number"),
                            "end_page_number": c.get("end_page_number", c.get("start_page_number")),
                        })
    return out


def ask_and_highlight_with_claude(
    pdf_path: str,
    out_pdf_path: str,
    question: str,
    settings: Settings,
    title: str | None = None,
    model: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """End-to-end: ask Claude with PDF, extract citations, highlight PDF.

    Returns (summary, citations_list). Requires ANTHROPIC_API_KEY.
    """
    from .pdf_highlighter import highlight_pdf  # local import to avoid cycles

    resp = claude_query_with_pdf_and_citations(pdf_path, question, settings, title=title, model=model)
    citations = extract_page_location_citations_from_claude(resp)
    # Even if citations empty, produce an output copy (no highlights)
    summary = highlight_pdf(pdf_path, out_pdf_path, citations)
    return summary, citations
