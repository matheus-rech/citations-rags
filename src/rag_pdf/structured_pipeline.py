from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json
import os
from openai import OpenAI
from jsonschema import validate
from .vision_analytics import VisionAnalyzer
from .config import load_settings
from .http_responses import responses_create_json_schema, extract_output_text
from .ocr_pipeline import ocr_linearize_pdf

# Schema tailored to user's meta-analysis requirements
META_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "study_metadata": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "first_author": {"type": "string"},
                "year": {"type": "integer"},
                "journal": {"type": "string"},
                "doi": {"type": "string"},
                "pmid": {"type": "string"},
            },
            "required": ["title", "first_author", "year", "journal", "doi", "pmid"],
            "additionalProperties": False,
        },
        "classification": {"type": "string", "enum": ["S", "PW"]},
        "pico": {
            "type": "object",
            "properties": {
                "population": {"type": "string"},
                "intervention": {"type": "string"},
                "comparator": {"type": "string"},
                "outcomes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["population", "intervention", "comparator", "outcomes"],
            "additionalProperties": False,
        },
        "sociodemographics": {
            "type": "object",
            "properties": {
                "n_total": {"type": "integer"},
                "age": {
                    "type": "object",
                    "properties": {
                        "mean": {"type": "number"},
                        "sd": {"type": "number"},
                        "median": {"type": "number"},
                        "iqr": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "sex": {
                    "type": "object",
                    "properties": {
                        "male_n": {"type": "integer"},
                        "female_n": {"type": "integer"},
                        "male_pct": {"type": "number"},
                        "female_pct": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "comorbidities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "n": {"type": "integer"},
                    "pct": {"type": "number"},
                    "provenance": {"type": "object"},
                },
                "required": ["name"],
                "additionalProperties": True,
            },
        },
        "indications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "provenance": {"type": "object"},
                },
                "required": ["text"],
                "additionalProperties": True,
            },
        },
        "imaging": {
            "type": "object",
            "properties": {
                "infarct_volume": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "arm_id": {"type": "string"},
                            "value": {"type": "number"},
                            "unit": {"type": "string"},
                            "method": {"type": "string"},
                            "provenance": {"type": "object"},
                        },
                        "required": ["value"],
                        "additionalProperties": True,
                    },
                },
                "peak_swelling": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "time": {"type": "string"},
                            "value": {"type": "number"},
                            "unit": {"type": "string"},
                            "provenance": {"type": "object"},
                        },
                        "required": ["time"],
                        "additionalProperties": True,
                    },
                },
            },
            "additionalProperties": False,
        },
        "arms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arm_id": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"type": "string", "enum": ["surgical", "conservative", "EVD", "single-arm", "other"]},
                    "n": {"type": "integer"},
                    "notes": {"type": "string"},
                    "provenance": {"type": "object"},
                },
                "required": ["arm_id", "label", "type"],
                "additionalProperties": True,
            },
        },
        "outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "definition": {"type": "string"},
                    "scale": {"type": "string"},
                    "timepoint": {"type": "string"},
                    "arm_id": {"type": "string"},
                    "comparison": {"type": "string"},
                    "statistic_type": {"type": "string"},
                    "value": {"type": "number"},
                    "dispersion_type": {"type": "string"},
                    "dispersion_value": {"type": "number"},
                    "n": {"type": "integer"},
                    "subgroup_label": {"type": "string"},
                    "sensitivity_label": {"type": "string"},
                    "quote": {"type": "string"},
                    "provenance": {"type": "object"},
                },
                "required": ["name", "timepoint"],
                "additionalProperties": True,
            },
        },
        "source_citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation_id": {"type": "string"},
                    "location": {"type": "string"},
                    "quote": {"type": "string"},
                    "start_offset": {"type": "integer"},
                    "end_offset": {"type": "integer"},
                    "page": {"type": "integer"},
                    "source": {"type": "string", "enum": ["text", "vision"]},
                    "offset_status": {"type": "string", "enum": ["matched", "unmatched"]},
                },
                "required": ["citation_id", "quote", "offset_status", "source"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "study_metadata",
        "classification",
        "pico",
        "arms",
        "outcomes",
        "source_citations",
    ],
    "additionalProperties": False,
}


def _load_parsed_doc_pages(filename: str) -> List[str]:
    """Load previously saved pages_description for a given filename from parsed JSON, if present."""
    s = load_settings()
    try:
        with open(s.parsed_json_path, 'r') as f:
            docs = json.load(f)
        for d in docs:
            if d.get('filename') == filename:
                return d.get('pages_description', []) or []
    except Exception:
        pass
    return []


def _find_offsets(text: str, quote: str) -> Optional[Tuple[int, int]]:
    if not text or not quote:
        return None
    idx = text.find(quote)
    if idx == -1:
        return None
    return idx, idx + len(quote)


def two_pass_provenance_resolution(data: Dict[str, Any], linear_text: str, filename: str) -> Dict[str, Any]:
    """For each quote-like field, try to match offsets in linear_text (pass 1),
    else scan per-page vision descriptions (pass 2) and add page/source flags.
    """
    pages = _load_parsed_doc_pages(filename)

    def resolve_quote(obj: Dict[str, Any]) -> Dict[str, Any]:
        quote = obj.get('quote') or ''
        res = {"offset_status": "unmatched", "source": "text", "start_offset": None, "end_offset": None, "page": None}
        m = _find_offsets(linear_text, quote)
        if m:
            res.update({"offset_status": "matched", "start_offset": m[0], "end_offset": m[1], "source": "text"})
            return res
        # Pass 2: search pages
        for i, ptxt in enumerate(pages, start=1):
            m2 = _find_offsets(ptxt, quote)
            if m2:
                return {"offset_status": "matched", "start_offset": m2[0], "end_offset": m2[1], "source": "vision", "page": i}
        # Unmatched
        return res

    # Update source_citations
    citations = data.get('source_citations', []) or []
    for c in citations:
        prov = resolve_quote(c)
        c.update(prov)

    # Update outcomes
    for o in data.get('outcomes', []) or []:
        if 'quote' in o and o['quote']:
            prov = resolve_quote(o)
            o['provenance'] = {**o.get('provenance', {}), **prov}

    # Update comorbidities/indications/imaging values if they have quotes (optional)
    return data


def extract_with_responses_api(text: str, model: str, api_key: str) -> Dict[str, Any]:
    """Request strictly structured JSON using Chat Completions with Structured Outputs.
    We use response_format=json_schema (strict) to have the model adhere to META_SCHEMA.
    """
    system = (
        "You are a meticulous extractor for systematic reviews/meta-analyses of decompressive suboccipital craniectomy (DC) in cerebellar stroke. "
        "Return ONLY valid JSON that strictly matches the requested schema. Do not include any additional commentary."
    )
    user = (
        "Review the provided text (linearized PDF and/or page descriptions). Focus on Methods/Results, tables and figures. "
        "Extract PICO elements, sociodemographics, indications, comorbidities, infarct volume and peak swelling, arms (S vs conservative/EVD), and outcomes (functional and mortality) with timepoints, statistics and dispersions. "
        "Classify as S (single-arm) or PW (pairwise or comparative). "
        "For each numeric value, include a short 'quote' verbatim if present."
        "\n\nJSON Schema:\n" + json.dumps(META_SCHEMA) + "\n\nTEXT BEGIN:\n" + text + "\nTEXT END."
    )
    # Call REST Responses API with strict json_schema
    resp_json = responses_create_json_schema(api_key=api_key, model=model, system=system, input_text=user, json_schema=META_SCHEMA)
    raw = extract_output_text(resp_json) or '{}'
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find('{')
        end = raw.rfind('}')
        data = json.loads(raw[start:end+1])
    try:
        validate(instance=data, schema=META_SCHEMA)
    except Exception:
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON parsing failed: {e}. Attempting to parse substring.")
        start = raw.find('{')
        end = raw.rfind('}')
        try:
            data = json.loads(raw[start:end+1])
        except json.JSONDecodeError as e2:
            logging.error(f"Fallback JSON parsing also failed: {e2}. Returning empty dict.")
            data = {}
    try:
        validate(instance=data, schema=META_SCHEMA)
    except Exception as e:
        logging.warning(f"Schema validation failed: {e}")
    return data


def run_strict_extraction_for_pdf(pdf_path: str, max_chars: int = 20000) -> Dict[str, Any]:
    s = load_settings()
    fname = os.path.basename(pdf_path)

    # Build linear text: prefer pdf text; if empty, fall back to GPT-4o page descriptions previously saved
    try:
        from .pdf_processing import extract_text_from_doc
        linear = extract_text_from_doc(pdf_path)
    except Exception:
        linear = ''

    # If no text, try OCR to build synthetic linear text
    if not linear:
        try:
            linear = ocr_linearize_pdf(pdf_path)
        except Exception:
            linear = ''

    # If still empty, fallback to GPT-4o page descriptions previously saved
    if not linear:
        pages = _load_parsed_doc_pages(fname)
        linear = '\n\n'.join(pages)

    text = linear[:max_chars] if linear else ''
    data = extract_with_responses_api(text=text, model=s.chat_model, api_key=s.openai_api_key)

    # Two-pass provenance resolution
    data = two_pass_provenance_resolution(data, linear_text=text, filename=fname)
    return data
