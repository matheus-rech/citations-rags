import json
from typing import Dict, Any
from openai import OpenAI
from jsonschema import validate

# JSON schema provided by the user
SCHEMA: Dict[str, Any] = {
    "name": "cerebellar_infarction_study_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "study_metadata": {
                "type": "object",
                "properties": {
                    "study_title": {"type": "string", "description": "Full title of the clinical study."},
                    "first_author": {"type": "string", "description": "First author of the publication."},
                    "year": {"type": "integer", "description": "Year the study was published."},
                    "journal": {"type": "string", "description": "Journal where the study appeared."},
                    "pmid": {"type": "string", "description": "PubMed ID, if available, of the article."},
                },
                "required": ["study_title", "first_author", "year", "journal", "pmid"],
                "additionalProperties": False,
            },
            "source_citations": {
                "type": "array",
                "description": "Referenceable spans (sections, pages, or sentences) mapped to 0-based offsets for precision.",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation_id": {"type": "string", "description": "Unique ID for this citation."},
                        "location": {"type": "string", "description": "Location (e.g., Results section, Table 1, or page reference)."},
                        "quote": {"type": "string", "description": "Exact quote or table cell from the text."},
                        "start_offset": {"type": "integer", "description": "0-based inclusive character offset for the start of the quote."},
                        "end_offset": {"type": "integer", "description": "0-based exclusive character offset for the end of the quote."},
                    },
                    "required": ["citation_id", "location", "quote", "start_offset", "end_offset"],
                    "additionalProperties": False,
                },
            },
            "cohort_groups": {
                "type": "array",
                "description": "Groupings used in the study, e.g., Surgical, Conservative, Control.",
                "items": {
                    "type": "object",
                    "properties": {
                        "cohort_group_id": {"type": "string", "description": "Unique identifier for the cohort group in this extraction."},
                        "label": {"type": "string", "description": "Label used in the publication (e.g. 'Surgical' or 'Conservative')."},
                        "in_context": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Verbatim text span describing the group."},
                                "start_offset": {"type": "integer", "description": "0-based inclusive character offset of the start of group text."},
                                "end_offset": {"type": "integer", "description": "0-based exclusive character offset of the end of group text."},
                            },
                            "required": ["text", "start_offset", "end_offset"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["cohort_group_id", "label", "in_context"],
                    "additionalProperties": False,
                },
            },
            "outcome_values": {
                "type": "array",
                "description": "Each entry is a measured outcome. There may be multiple entries for different cohorts, timepoints, or statistics.",
                "items": {
                    "type": "object",
                    "properties": {
                        "outcome_name": {"type": "string", "description": "Name/description of the outcome (e.g., mRS 0-2, mortality, GCS at admission, hydrocephalus)."},
                        "outcome_type": {"type": "string", "description": "Type of outcome (e.g., functional, radiographic, surgical, complication, death)."},
                        "cohort_group_id": {"type": "string", "description": "cohort_group_id to which this outcome pertains."},
                        "timepoint": {"type": "string", "description": "Timepoint for the outcome (e.g., at discharge, 6 months, 12 months)."},
                        "statistic": {"type": "string", "description": "Statistic reported (e.g., mean, median, count, percent, OR)."},
                        "value": {"type": "string", "description": "Reported value as per publication."},
                        "unit": {"type": "string", "description": "Unit, if applicable."},
                        "citation_ids": {"type": "array", "description": "Array of unique citation IDs relevant to this value.", "items": {"type": "string"}},
                        "in_context": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Verbatim text span for this outcome value."},
                                "start_offset": {"type": "integer", "description": "0-based inclusive character offset of start in the source text."},
                                "end_offset": {"type": "integer", "description": "0-based exclusive character offset of end in the source text."},
                            },
                            "required": ["text", "start_offset", "end_offset"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "outcome_name",
                        "outcome_type",
                        "cohort_group_id",
                        "timepoint",
                        "statistic",
                        "value",
                        "unit",
                        "citation_ids",
                        "in_context",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["study_metadata", "source_citations", "cohort_groups", "outcome_values"],
        "additionalProperties": False,
    },
}


SYSTEM_INSTRUCTIONS = (
    "You are a meticulous clinical NLP extractor. Given the full linearized text of a study, "
    "produce a single JSON object that STRICTLY conforms to the provided JSON schema. "
    "- Use 0-based character offsets based on the exact text provided in this prompt.\n"
    "- Quotes must be verbatim from the text.\n"
    "- If a required field is unknown or not present, use an empty string (or 0 for integers).\n"
    "- Do not add extra fields.\n"
    "- Be conservative; only extract facts supported by the text.\n"
)


def extract_structured_from_text(api_key: str, model: str, text: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    # Use chat.completions with function-like instructions for JSON only
    sys = SYSTEM_INSTRUCTIONS + "\nReturn only a valid JSON object matching the schema, no extra text."
    prompt = (
        "You will receive the full study text below. Extract and return a JSON object strictly matching the schema.\n\n"
        + text
    )
    comp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    raw = comp.choices[0].message.content or "{}"

    # Robust JSON parsing: direct, or extract braces span
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise

    # Validate against schema (best-effort)
    try:
        validate(instance=data, schema=SCHEMA["schema"])  # validate against the JSON schema part
    except Exception:
        pass
    return data
