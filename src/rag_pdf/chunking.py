import re
from typing import List, Dict, Any


def merge_and_chunk(docs: List[Dict[str, Any]], remove_first_page: bool = True) -> List[Dict[str, str]]:
    """
    Mimics the notebook logic:
    - Split extracted text by form feed (page separator) and optionally drop first page
    - Combine slide text with a matching vision description (matching by identical first line case-insensitive)
    - Append unmatched vision descriptions
    - Returns a list of dicts, each with "content" and "filename"
    """
    chunks: List[Dict[str, str]] = []

    for doc in docs:
        filename = doc.get("filename", "unknown")
        text_pages = doc.get("text", "").split('\f')
        descriptions = doc.get("pages_description", [])

        if remove_first_page and len(text_pages) > 0:
            text_pages = text_pages[1:]
            if len(descriptions) > 0:
                descriptions = descriptions[1:]

        used_desc_idx = set()

        for page in text_pages:
            page = page or ""
            slide_content = page + "\n"
            slide_title = (page.split('\n')[0] or "").strip()
            for j, desc in enumerate(descriptions):
                desc_title = (desc.split('\n')[0] or "").strip()
                if slide_title and slide_title.lower() == desc_title.lower():
                    cleaned_desc = desc[len(desc_title):].lstrip("\n") if desc_title else desc
                    slide_content += cleaned_desc
                    used_desc_idx.add(j)
            chunks.append({"content": slide_content, "filename": filename})

        for j, desc in enumerate(descriptions):
            if j not in used_desc_idx:
                chunks.append({"content": desc, "filename": filename})

    return chunks


def clean_content(pieces: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    for piece in pieces:
        content = piece["content"]
        filename = piece["filename"]

        lines = [line.strip() for line in content.strip().split('\n')]
        lines = [line for line in lines if line and not re.fullmatch(r"\d{1,2}", line)]

        text = '\n'.join(lines)
        text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r" +", " ", text)
        text = re.sub(r'\n{2,}', '\n', text)

        cleaned.append({"content": text.strip(), "filename": filename})
    return cleaned
