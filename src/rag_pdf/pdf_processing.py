from typing import List
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from PIL import Image
import fitz  # PyMuPDF as a robust fallback for text & images
import io


def convert_doc_to_images(path: str) -> List[Image.Image]:
    # Try pdf2image (requires poppler system dependency)
    try:
        images = convert_from_path(path)
        if images:
            return images
    except Exception:
        pass

    # Fallback to PyMuPDF rendering
    images: List[Image.Image] = []
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    return images


def extract_text_from_doc(path: str) -> str:
    # Try pdfminer first
    try:
        text = extract_text(path)
        if text and text.strip():
            return text
    except Exception:
        pass
    # Fallback to PyMuPDF
    try:
        doc = fitz.open(path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "\f".join(parts)
    except Exception:
        return ""
