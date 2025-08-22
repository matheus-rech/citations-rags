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
    entity_type: str | None = None  # e.g., "variable", "method", "result", "conclusion"
    color: str | None = None  # e.g., "yellow", "green", "blue", "red", "purple"

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
        # Extract entity type and color if available
        entity_type = obj.get("entity_type") or obj.get("type") or obj.get("category")
        color = obj.get("color") or obj.get("highlight_color")
        return cls(
            cited_text=str(cited_text or ""),
            start_page_number=int(sp),
            end_page_number=int(ep),
            entity_type=entity_type,
            color=color
        )


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


# Color mapping for different entity types
ENTITY_COLORS = {
    "variable": (1, 1, 0),      # yellow
    "method": (0, 1, 0),       # green
    "result": (0, 0.5, 1),     # blue
    "conclusion": (1, 0.5, 0), # orange
    "data": (1, 0, 1),         # magenta
    "reference": (0.5, 0.5, 0.5), # gray
    "default": (1, 1, 0),      # yellow (fallback)
}

COLOR_NAMES = {
    "yellow": (1, 1, 0),
    "green": (0, 1, 0),
    "blue": (0, 0.5, 1),
    "red": (1, 0, 0),
    "purple": (0.5, 0, 1),
    "orange": (1, 0.5, 0),
    "magenta": (1, 0, 1),
    "cyan": (0, 1, 1),
    "gray": (0.5, 0.5, 0.5),
}

def _get_color_for_citation(citation: PageLocationCitation) -> tuple:
    """Get RGB color tuple for a citation based on its color or entity_type."""
    if citation.color and citation.color.lower() in COLOR_NAMES:
        return COLOR_NAMES[citation.color.lower()]
    if citation.entity_type and citation.entity_type.lower() in ENTITY_COLORS:
        return ENTITY_COLORS[citation.entity_type.lower()]
    return ENTITY_COLORS["default"]

def _search_and_highlight(page: fitz.Page, text: str, citation: PageLocationCitation) -> int:
    """Search the given page for the snippet and add highlights around matches.
    Returns number of rectangles highlighted.
    """
    try:
        rects = page.search_for(text)
    except Exception:
        rects = []
    
    color = _get_color_for_citation(citation)
    count = 0
    
    for r in rects:
        try:
            annot = page.add_highlight_annot(r)
            annot.set_colors({"stroke": color})
            # Add a note with entity type and snippet preview
            if citation.entity_type or len(citation.cited_text) > 50:
                note_text = ""
                if citation.entity_type:
                    note_text += f"[{citation.entity_type.upper()}] "
                note_text += citation.cited_text[:100]
                if len(citation.cited_text) > 100:
                    note_text += "..."
                annot.set_info(content=note_text)
            annot.update()
            count += 1
        except Exception:
            # Ignore annotation failures for odd rects
            pass
    return count

def _fuzzy_rectangle_highlight(page: fitz.Page, text: str, citation: PageLocationCitation) -> int:
    """Fallback: try to find text using block-level search and draw rectangles.
    Returns number of rectangles drawn.
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return 0
    
    color = _get_color_for_citation(citation)
    norm_target = _normalize_text(text).lower()
    count = 0
    
    if len(norm_target) < 10:  # Skip very short text for fuzzy matching
        return 0
    
    # Extract words from target text
    target_words = [w for w in re.findall(r"\w+", norm_target) if len(w) > 3]
    if len(target_words) < 2:
        return 0
    
    for block in blocks:
        if "lines" not in block:
            continue
        
        # Build block text and check for word overlap
        block_text = ""
        block_bbox = None
        
        for line in block["lines"]:
            for span in line.get("spans", []):
                block_text += span.get("text", "") + " "
                if block_bbox is None:
                    block_bbox = span["bbox"]
                else:
                    # Expand bounding box
                    x0, y0, x1, y1 = block_bbox
                    sx0, sy0, sx1, sy1 = span["bbox"]
                    block_bbox = (min(x0, sx0), min(y0, sy0), max(x1, sx1), max(y1, sy1))
        
        if block_bbox is None:
            continue
        
        block_norm = _normalize_text(block_text).lower()
        
        # Check if enough target words are present
        matches = sum(1 for word in target_words if word in block_norm)
        if matches >= min(3, len(target_words) * 0.6):  # At least 60% of words or 3 words
            try:
                rect = fitz.Rect(block_bbox)
                annot = page.add_rect_annot(rect)
                annot.set_colors({"stroke": color, "fill": (*color, 0.2)})  # Semi-transparent fill
                annot.set_border(width=2)
                # Add note indicating fuzzy match
                note_text = f"[FUZZY MATCH]"
                if citation.entity_type:
                    note_text += f" [{citation.entity_type.upper()}]"
                note_text += f" {citation.cited_text[:100]}"
                if len(citation.cited_text) > 100:
                    note_text += "..."
                annot.set_info(content=note_text)
                annot.update()
                count += 1
            except Exception:
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
            found_exact_match = False
            
            # Try exact text search first
            for snippet in _candidate_snippets(cit.cited_text):
                hits = _search_and_highlight(page, snippet, cit)
                hit_this_page += hits
                if hits:
                    found_exact_match = True
                    # If we got a full-length match, no need to try more variants
                    if len(snippet) >= min(200, len(_normalize_text(cit.cited_text))):
                        break
            
            # If no exact matches found, try fuzzy rectangle matching
            if not found_exact_match and len(_normalize_text(cit.cited_text)) > 20:
                fuzzy_hits = _fuzzy_rectangle_highlight(page, cit.cited_text, cit)
                hit_this_page += fuzzy_hits
            
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
        try:
            data = json.loads(path_or_json)
        except json.JSONDecodeError as e:
            raise ValueError("Input is neither a valid file path nor a valid JSON string.") from e
    if isinstance(data, dict) and "citations" in data:
        return data["citations"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported citations json format. Expect list or object with 'citations'.")
