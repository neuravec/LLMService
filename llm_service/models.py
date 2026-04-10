"""
Model registry — knows which Azure OpenAI models are reasoning models
and what parameters they support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Capability flags per model family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCapabilities:
    """What a given model family supports."""
    reasoning: bool = False              # uses reasoning_effort instead of temperature
    supports_temperature: bool = True
    supports_system_message: bool = True
    supports_stop: bool = True           # o3/o4-mini do NOT support stop
    default_api_version: str = "2025-04-01-preview"


# ---------------------------------------------------------------------------
# Known model families — extend as new models appear
# ---------------------------------------------------------------------------

_FAMILIES: list[tuple[re.Pattern, ModelCapabilities]] = [
    # o-series reasoning models
    (re.compile(r"^o[134]"), ModelCapabilities(
        reasoning=True,
        supports_temperature=False,
        supports_system_message=False,
        supports_stop=False,
    )),
    # GPT-5.x reasoning family
    (re.compile(r"^gpt-5"), ModelCapabilities(
        reasoning=True,
        supports_temperature=False,
        supports_system_message=True,
        supports_stop=True,
    )),
    # GPT-4.1 and similar standard models
    (re.compile(r"^gpt-4"), ModelCapabilities(
        reasoning=False,
        supports_temperature=True,
        supports_system_message=True,
        supports_stop=True,
    )),
]

# Fallback for unknown models — treat as standard
_DEFAULT = ModelCapabilities()


def detect_capabilities(model_name: str) -> ModelCapabilities:
    """Match *model_name* against known families. Returns safe defaults
    for unknown models (standard, non-reasoning)."""
    name = model_name.lower().strip()
    for pattern, caps in _FAMILIES:
        if pattern.search(name):
            return caps
    return _DEFAULT
