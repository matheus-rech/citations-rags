from __future__ import annotations
import os
import json
import httpx
from typing import Dict, Any, Optional

OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def responses_create_json_schema(api_key: str, model: str, system: str, input_text: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{OPENAI_BASE}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "system": system,
        "input": input_text,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
            "strict": True,
        },
        "max_output_tokens": 4000,
    }
    with httpx.Client(timeout=120) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


def extract_output_text(resp_json: Dict[str, Any]) -> str:
    # Try std fields first
    if "output_text" in resp_json and resp_json["output_text"]:
        return resp_json["output_text"]
    # Walk nested structure
    try:
        return resp_json["output"][0]["content"][0]["text"]
    except Exception:
        pass
    # Fallback
    return ""
