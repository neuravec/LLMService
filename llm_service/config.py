"""
Configuration loader — YAML file → validated settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class LLMConfig:
    """Minimal required + optional settings for Azure OpenAI calls.

    Only three fields are mandatory: api_key, endpoint, model_name.
    Everything else has sensible defaults.
    """

    # --- required ---
    api_key: str
    endpoint: str          # e.g. https://<resource>.openai.azure.com
    model_name: str        # deployment name in Azure

    # --- optional ---
    api_version: str = ""                  # auto-detected from model if empty
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    concurrency: int = 8                   # semaphore limit
    timeout: float = 120.0                 # seconds per request
    retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)  # pass-through params

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMConfig":
        """Load config from a YAML file.

        The YAML can use env-var placeholders: ``${ENV_VAR}`` or
        ``${ENV_VAR:default_value}``.
        Unknown keys are silently ignored.
        """
        text = Path(path).read_text(encoding="utf-8")
        text = _resolve_env_vars(text)
        data: dict = yaml.safe_load(text) or {}
        return cls(**_filter_known_fields(cls, data))

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        return cls(**_filter_known_fields(cls, d))


# ---------------------------------------------------------------------------
# Env-var substitution helpers
# ---------------------------------------------------------------------------

import re as _re

_ENV_PATTERN = _re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _filter_known_fields(cls: type, data: dict) -> dict:
    """Keep only keys that match dataclass fields, drop None values."""
    import dataclasses
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in valid and v is not None}


def _resolve_env_vars(text: str) -> str:
    def _replace(m):
        name, default = m.group(1), m.group(2)
        return os.environ.get(name, default if default is not None else m.group(0))
    return _ENV_PATTERN.sub(_replace, text)
