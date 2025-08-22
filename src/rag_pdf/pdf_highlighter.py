from __future__ import annotations
"""
PDF citation highlighting utilities using PyMuPDF (fitz).

This module accepts a list of citation objects that include the cited text and
1-based page numbers, searches the PDF for those spans, and writes a new PDF
with highlight annotations drawn around the matched text.

It is designed to work with citation payloads similar to Anthropic/Claude's
`page_location` citations, but is provider-agnostic. You can pass your own
citation list as long as fields are named similarly.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable
import re
import json
import fitz  # PyMuPDF


@dataclass
class PageLocationCitation:
    cited_text: str
    start_page_number: int  # 1-based inclusive
    end_page_number: int | None = None  # 1-based inclusive (optional)

    @classmethod
    def from_generic(cls, obj: Dict[str, Any]) -> "PageLocationCitation":
        # Accept various shapes commonly returned by citation APIs
        cited_text = obj.get("cited_text") or obj.get("text") or ""
        # Some providers may use 'page_number' for single-page citations
        sp = obj.get("start_page_number") or obj.get("page_number") or obj.get("page")
        ep = obj.get("end_page_number")
        if sp is None:
            # Fallback: if 'page_location' nested
            loc = obj.get("page_location") or {}
            sp = loc.get("start_page_number") or loc.get("page_number")
            ep = loc.get("end_page_number")
        if sp is None:
            raise ValueError(f"Citation missing page numbers: {obj}")
        if ep is None:
            ep = sp
        return cls(cited_text=str(cited_text or ""), start_page_number=int(sp), end_page_number=int(ep))


def _normalize_text(t: str) -> str:
    # Remove control chars like \u0002 that may appear in citations
    t = t.replace("\u0002", " ")
    # Collapse whitespace and hyphenation across newlines commonly found in PDFs
    t = re.sub(r"-\s*\n", "", t)  # fix hyphenated line breaks
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _candidate_snippets(full: str) -> List[str]:
    """Generate a set of search candidates from the cited text.
    We try exact, shortened, and robust variants to improve hit rate.
    """
    candidates: List[str] = []
    if not full:
        return candidates
    norm = _normalize_text(full)
    if not norm:
        return candidates
    candidates.append(norm)
    # If text is very long, also try a head and a mid slice
    if len(norm) > 300:
        candidates.append(norm[:200])
        mid = norm[len(norm)//4 : len(norm)//4 + 200]
        if mid not in candidates:
            candidates.append(mid)
    # Try removing quotes or surrounding punctuation
    stripped = norm.strip("\"'“”‘’()[]{}")
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    # Try a phrase built from the longest words
    words = [w for w in re.findall(r"\w[\w%:/.-]*", norm) if len(w) > 4]
    if words:
        words = sorted(words, key=len, reverse=True)[:6]
        phrase = " ".join(words[:4])
        if len(phrase) > 20 and phrase not in candidates:
            candidates.append(phrase)
    return candidates


def _search_and_highlight(page: fitz.Page, text: str) -> int:
    """Search the given page for the snippet and add highlights around matches.
    Returns number of rectangles highlighted.
    """
    try:
        rects = page.search_for(text)
    except Exception:
        rects = []
    count = 0
    for r in rects:
        try:
            annot = page.add_highlight_annot(r)
            annot.set_colors({"stroke": (1, 1, 0)})  # yellow
            annot.update()
            count += 1
        except Exception:
            # Ignore annotation failures for odd rects
            pass
    return count


def highlight_pdf(pdf_path: str, output_pdf_path: str, citations: Iterable[Dict[str, Any] | PageLocationCitation]) -> Dict[str, Any]:
    """Create a highlighted copy of the PDF given citations.

    Args:
        pdf_path: input PDF path
        output_pdf_path: where to save the highlighted PDF
        citations: iterable of dicts or PageLocationCitation items. Must include
                   fields like 'cited_text', 'start_page_number', 'end_page_number' (1-based)

    Returns:
        Summary dict with counts of matches per-page and overall.
    """
    # Normalize input
    norm_citations: List[PageLocationCitation] = []
    for c in citations:
        if isinstance(c, PageLocationCitation):
            norm_citations.append(c)
        else:
            norm_citations.append(PageLocationCitation.from_generic(c))

    doc = fitz.open(pdf_path)
    per_page_hits: Dict[int, int] = {}

    for cit in norm_citations:
        start_idx = max(0, cit.start_page_number - 1)
        end_idx = min(len(doc) - 1, (cit.end_page_number or cit.start_page_number) - 1)
        for page_num in range(start_idx, end_idx + 1):
            page = doc[page_num]
            hit_this_page = 0
            for snippet in _candidate_snippets(cit.cited_text):
                hits = _search_and_highlight(page, snippet)
                hit_this_page += hits
                if hits:
                    # If we got a full-length match, no need to try more variants
                    if len(snippet) >= min(200, len(_normalize_text(cit.cited_text))):
                        break
            per_page_hits[page_num + 1] = per_page_hits.get(page_num + 1, 0) + hit_this_page

    doc.save(output_pdf_path)
    doc.close()

    total = sum(per_page_hits.values())
    return {"total_highlights": total, "per_page": per_page_hits}


def load_citations_from_json(path_or_json: str) -> List[Dict[str, Any]]:
    """Load citation list from a file path or an inline JSON string.
    Supports shapes: {"citations": [...]} or just a list [...].
    """
    try:
        # First try to treat as a file path
        with open(path_or_json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = json.loads(path_or_json)
    if isinstance(data, dict) and "citations" in data:
        return data["citations"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported citations json format. Expect list or object with 'citations'.")
