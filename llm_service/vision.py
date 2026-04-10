"""
Vision / multimodal helpers — encode images for Azure OpenAI.

Supports:
  - Local file paths (auto base64 encode)
  - URLs (passed through as-is)
  - Raw bytes (base64 encode with explicit media type)
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Union

# Types that can represent an image
ImageInput = Union[str, Path, bytes]


def encode_image(source: ImageInput, media_type: str | None = None) -> dict[str, Any]:
    """Convert an image source to Azure OpenAI image_url content part.

    Args:
        source: File path (str/Path), URL (str starting with http), or raw bytes.
        media_type: MIME type override (e.g. "image/png"). Auto-detected for files.

    Returns:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    if isinstance(source, bytes):
        mt = media_type or "image/png"
        b64 = base64.b64encode(source).decode("ascii")
        return _image_part(f"data:{mt};base64,{b64}")

    source_str = str(source)

    # URL — pass through
    if source_str.startswith(("http://", "https://")):
        return _image_part(source_str)

    # Local file path
    path = Path(source_str)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mt = media_type or _detect_media_type(path)
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return _image_part(f"data:{mt};base64,{b64}")


def build_content_parts(
    text: str,
    images: list[ImageInput] | None = None,
    image_detail: str = "auto",
) -> str | list[dict[str, Any]]:
    """Build the 'content' field for a user message.

    If no images, returns plain string (cheaper, simpler).
    If images present, returns content array with text + image_url parts.

    Args:
        text: The text portion of the message.
        images: List of image sources (paths, URLs, bytes).
        image_detail: "auto" | "low" | "high" — resolution hint for Azure.
    """
    if not images:
        return text

    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        part = encode_image(img)
        # Add detail level
        part["image_url"]["detail"] = image_detail
        parts.append(part)
    return parts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _detect_media_type(path: Path) -> str:
    """Detect MIME type from file extension."""
    ext = path.suffix.lower()
    mt = _MIME_MAP.get(ext)
    if mt:
        return mt
    # Fallback to mimetypes module
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"
