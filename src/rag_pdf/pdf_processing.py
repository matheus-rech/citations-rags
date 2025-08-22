from typing import List
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from PIL import Image


def convert_doc_to_images(path: str) -> List[Image.Image]:
    # Note: requires poppler to be installed in system for pdf2image
    images = convert_from_path(path)
    return images


def extract_text_from_doc(path: str) -> str:
    try:
        text = extract_text(path)
    except Exception:
        text = ""
    return text
