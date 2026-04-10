"""
llm_service — universal async Azure OpenAI toolkit.

Quick start::

    from llm_service import LLMConfig, LLMClient

    cfg = LLMConfig.from_yaml("config.yaml")
    async with LLMClient(cfg) as llm:
        answer = await llm.chat("Summarize this document", system="You extract key facts.")
"""

from .config import LLMConfig
from .models import ModelCapabilities, detect_capabilities
from .client import LLMClient, LLMError
from .structured import parse_response, parse_response_lenient
from .pipeline import Step, Pipeline, PipelineResult, StepResult
from .usage import UsageTracker, RequestUsage, get_pricing
from .vision import encode_image, build_content_parts, pdf_to_images, ImageInput

__all__ = [
    "LLMConfig",
    "LLMClient",
    "LLMError",
    "ModelCapabilities",
    "detect_capabilities",
    "parse_response",
    "parse_response_lenient",
    "Step",
    "Pipeline",
    "PipelineResult",
    "StepResult",
    "UsageTracker",
    "RequestUsage",
    "get_pricing",
    "encode_image",
    "build_content_parts",
    "pdf_to_images",
    "ImageInput",
]
