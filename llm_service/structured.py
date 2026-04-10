"""
Structured output — parse LLM responses into Pydantic models.

Supports two modes:
  1. json_object response_format  — LLM returns raw JSON, we validate with Pydantic.
  2. json_schema response_format  — sends Pydantic schema to Azure so the model
     is constrained to produce valid JSON (Structured Outputs API).

Both are exposed through LLMClient.structured() for maximum convenience.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("llm_service")

T = TypeVar("T", bound=BaseModel)


def pydantic_to_json_schema(model: type[T]) -> dict[str, Any]:
    """Convert a Pydantic model class to the JSON schema dict
    expected by Azure OpenAI Structured Outputs.

    Wraps the schema in the envelope:
      { "type": "json_schema", "json_schema": { "name": ..., "strict": true, "schema": ... } }
    """
    raw = model.model_json_schema()
    # Azure requires all properties to be required and additionalProperties: false
    _enforce_strict(raw)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": raw,
        },
    }


def response_format_json() -> dict[str, str]:
    """Simple JSON-object mode (no schema enforcement)."""
    return {"type": "json_object"}


def parse_response(text: str, model: type[T]) -> T:
    """Parse raw LLM text into a Pydantic model instance.

    Handles common LLM quirks:
      - ```json ... ``` fencing
      - Leading/trailing whitespace
    """
    cleaned = _strip_json_fences(text)
    data = json.loads(cleaned)
    return model.model_validate(data)


def parse_response_lenient(text: str, model: type[T]) -> T | dict[str, Any]:
    """Try Pydantic parse; on failure return raw dict (never raises)."""
    cleaned = _strip_json_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Response is not valid JSON, returning raw text")
        return {"_raw": text}
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        logger.warning("Pydantic validation failed: %s — returning raw dict", exc)
        return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        # Remove first line (```json) and last line (```)
        lines = t.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        t = "\n".join(lines)
    return t.strip()


def _enforce_strict(schema: dict) -> None:
    """Recursively set additionalProperties: false and make all props required.
    This is needed for Azure Structured Outputs strict mode."""
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema.setdefault("required", list(schema["properties"].keys()))
    # Recurse into nested schemas
    for key in ("properties", "$defs"):
        container = schema.get(key, {})
        for v in container.values():
            if isinstance(v, dict):
                _enforce_strict(v)
    for key in ("items", "anyOf", "oneOf", "allOf"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            _enforce_strict(sub)
        elif isinstance(sub, list):
            for item in sub:
                if isinstance(item, dict):
                    _enforce_strict(item)
