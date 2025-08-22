import re
from typing import List, Dict, Any


def merge_and_chunk(docs: List[Dict[str, Any]], remove_first_page: bool = True) -> List[str]:
    """
    Mimics the notebook logic:
    - Split extracted text by form feed (page separator) and optionally drop first page
    - Combine slide text with a matching vision description (matching by identical first line case-insensitive)
    - Append unmatched vision descriptions
    """
    content: List[str] = []

    for doc in docs:
        text_pages = doc.get("text", "").split('\f')
        if remove_first_page and len(text_pages) > 0:
            text_pages = text_pages[1:]
        descriptions = doc.get("pages_description", [])
        used_desc_idx = set()

        for page in text_pages:
            page = page or ""
            slide_content = page + "\n"
            slide_title = (page.split('\n')[0] or "").strip()
            for j, desc in enumerate(descriptions):
                desc_title = (desc.split('\n')[0] or "").strip()
                if slide_title and slide_title.lower() == desc_title.lower():
                    # remove title from desc, then append
                    cleaned_desc = desc[len(desc_title):].lstrip("\n") if desc_title else desc
                    slide_content += cleaned_desc
                    used_desc_idx.add(j)
            content.append(slide_content)

        # add unmatched descriptions
        for j, desc in enumerate(descriptions):
            if j not in used_desc_idx:
                content.append(desc)

    return content


def clean_content(pieces: List[str]) -> List[str]:
    cleaned = []
    for c in pieces:
        text = c.replace(' \n', '').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
        text = re.sub(r"(?<=\n)\d{1,2}", "", text)
        text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
        cleaned.append(text)
    return cleaned
