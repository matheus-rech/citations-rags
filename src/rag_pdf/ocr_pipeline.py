from __future__ import annotations
from typing import List
from .pdf_processing import convert_doc_to_images
from .vision_analytics import VisionAnalyzer
from .config import load_settings


def ocr_linearize_pdf(pdf_path: str) -> str:
    """Render pages and OCR with GPT-4o to build a synthetic linear text."""
    s = load_settings()
    analyzer = VisionAnalyzer(api_key=s.openai_api_key, model=s.chat_model)
    images = convert_doc_to_images(pdf_path)
    # Keep all pages (including first) for OCR completeness
    ocr_pages: List[str] = []
    for img in images:
        try:
            txt = analyzer.ocr_image(img)
        except Exception:
            txt = ""
        ocr_pages.append(txt or "")
    return "\n\n".join(ocr_pages)
