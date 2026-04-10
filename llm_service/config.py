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
    """Configuration for Azure OpenAI API connection.

    Only three fields are required: ``api_key``, ``endpoint``, ``model_name``.
    Everything else is optional with sensible defaults.

    Args:
        api_key: Azure OpenAI API key. Required.
        endpoint: Azure resource endpoint, e.g. ``"https://my-resource.openai.azure.com"``. Required.
        model_name: Deployment name in Azure (e.g. ``"gpt-4.1"``). Required.
        api_version: API version string. Auto-detected from model if empty.
        temperature: Sampling temperature 0-2. Ignored for reasoning models.
        max_tokens: Max output tokens. Sent as ``max_completion_tokens`` to Azure.
        reasoning_effort: For reasoning models: ``"low"`` | ``"medium"`` | ``"high"``.
            GPT-5.x also supports ``"none"`` | ``"minimal"`` | ``"xhigh"``.
        top_p: Nucleus sampling 0-1. Ignored for reasoning models.
        frequency_penalty: Frequency penalty -2.0 to 2.0. Ignored for reasoning models.
        presence_penalty: Presence penalty -2.0 to 2.0. Ignored for reasoning models.
        stop: Up to 4 stop sequences. Ignored for o3/o4-mini.
        concurrency: Max parallel requests (semaphore size). Default ``8``.
        timeout: Seconds per request before timeout. Default ``120``.
        retries: Max retry attempts on transient errors. Default ``3``.
        extra: Dict of additional params passed through to the API body
            (e.g. ``{"seed": 42, "user": "pipeline-v2"}``).

    Examples:
        From YAML file::

            cfg = LLMConfig.from_yaml("config.yaml")

        From code::

            cfg = LLMConfig(
                api_key=os.environ["AZURE_OPENAI_KEY"],
                endpoint="https://my-resource.openai.azure.com",
                model_name="gpt-4.1",
                temperature=0.3,
            )

        From dict (e.g. loaded from JSON)::

            cfg = LLMConfig.from_dict({"api_key": "...", "endpoint": "...", "model_name": "gpt-4.1"})
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

        Supports ``${ENV_VAR}`` and ``${ENV_VAR:default}`` substitution.
        Unknown keys in the YAML are silently ignored.

        Args:
            path: Path to the YAML config file.

        Returns:
            LLMConfig: Validated configuration instance.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            TypeError: If required fields (api_key, endpoint, model_name) are missing.

        Example::

            # config.yaml:
            # api_key: ${AZURE_OPENAI_KEY}
            # endpoint: https://my-resource.openai.azure.com
            # model_name: gpt-4.1
            # temperature: 0.3

            cfg = LLMConfig.from_yaml("config.yaml")
        """
        text = Path(path).read_text(encoding="utf-8")
        text = _resolve_env_vars(text)
        data: dict = yaml.safe_load(text) or {}
        return cls(**_filter_known_fields(cls, data))

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        """Create config from a dictionary. Unknown keys are silently ignored.

        Args:
            d: Dictionary with config values.

        Returns:
            LLMConfig: Validated configuration instance.

        Example::

            cfg = LLMConfig.from_dict({
                "api_key": "sk-...",
                "endpoint": "https://my-resource.openai.azure.com",
                "model_name": "gpt-4.1",
            })
        """
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
