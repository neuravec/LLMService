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
    """Capability flags for a model family.

    Attributes:
        reasoning: If ``True``, model uses ``reasoning_effort`` instead of ``temperature``.
        supports_temperature: If ``False``, ``temperature``/``top_p``/penalties are stripped.
        supports_system_message: If ``False``, system messages are sent as ``"developer"`` role.
        supports_stop: If ``False``, ``stop`` sequences are stripped from the request.
        default_api_version: Azure API version used if not specified in config.
    """
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
    """Detect model capabilities from its name.

    Matches against known model families (o-series, gpt-5.x, gpt-4.x).
    Unknown models get safe defaults (standard, non-reasoning).

    Args:
        model_name: Azure deployment or model name (e.g. ``"gpt-4.1"``, ``"o3"``).

    Returns:
        ModelCapabilities: Frozen dataclass with capability flags.

    Example::

        from llm_service import detect_capabilities

        caps = detect_capabilities("gpt-4.1")
        caps.reasoning           # False
        caps.supports_temperature  # True

        caps = detect_capabilities("o4-mini")
        caps.reasoning           # True
        caps.supports_stop       # False
    """
    name = model_name.lower().strip()
    for pattern, caps in _FAMILIES:
        if pattern.search(name):
            return caps
    return _DEFAULT
