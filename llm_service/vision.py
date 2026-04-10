"""
Vision / multimodal helpers — encode images and PDFs for Azure OpenAI.

Supports:
  - Local file paths: images (jpg, png, webp, ...) and PDFs (auto page-to-image)
  - URLs (passed through as-is)
  - Raw bytes (base64 encode with explicit media type)
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Sequence, Union

logger = logging.getLogger("llm_service")

# Types that can represent an image or PDF
ImageInput = Union[str, Path, bytes]


def encode_image(source: ImageInput, media_type: str | None = None) -> dict[str, Any]:
    """Convert an image source to Azure OpenAI ``image_url`` content part.

    Args:
        source: One of:

            - **File path** (``str`` or ``Path``) — auto-detects MIME type from extension,
              reads file and encodes as base64 data URI.
            - **URL** (``str`` starting with ``http://`` or ``https://``) — passed through as-is.
            - **Raw bytes** — encoded as base64 with ``media_type`` (default ``image/png``).

        media_type: MIME type override (e.g. ``"image/jpeg"``). Auto-detected for file paths.

    Returns:
        dict: ``{"type": "image_url", "image_url": {"url": "..."}}``.

    Raises:
        FileNotFoundError: If a file path doesn't exist.

    Example::

        from llm_service import encode_image

        # From file
        part = encode_image("photo.jpg")

        # From URL
        part = encode_image("https://example.com/img.png")

        # From bytes
        part = encode_image(raw_bytes, media_type="image/jpeg")
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
        raise FileNotFoundError(f"File not found: {path}")

    mt = media_type or _detect_media_type(path)
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return _image_part(f"data:{mt};base64,{b64}")


def pdf_to_images(
    path: str | Path,
    pages: Sequence[int] | None = None,
    dpi: int = 200,
) -> list[bytes]:
    """Render PDF pages to PNG images.

    Requires ``pymupdf`` (``pip install pymupdf``).

    Args:
        path: Path to the PDF file.
        pages: List of 0-based page indices to render.
            ``None`` (default) = all pages.
        dpi: Resolution for rendering. Default ``200`` — good balance
            between OCR quality and token cost. Use ``300`` for small text.

    Returns:
        list[bytes]: List of PNG image bytes, one per page.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ImportError: If ``pymupdf`` is not installed.

    Example::

        from llm_service.vision import pdf_to_images

        # All pages
        page_images = pdf_to_images("invoice.pdf")

        # Only first 2 pages, high quality
        page_images = pdf_to_images("report.pdf", pages=[0, 1], dpi=300)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF support. Install it with: pip install pymupdf"
        )

    result: list[bytes] = []
    with pymupdf.open(str(path)) as doc:
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        page_indices = pages if pages is not None else range(len(doc))
        for idx in page_indices:
            if idx < 0 or idx >= len(doc):
                logger.warning("PDF page %d out of range (0-%d), skipping", idx, len(doc) - 1)
                continue
            pix = doc[idx].get_pixmap(matrix=matrix)
            result.append(pix.tobytes("png"))

    logger.info("Rendered %d pages from %s at %d DPI", len(result), path.name, dpi)
    return result


def build_content_parts(
    text: str,
    images: list[ImageInput] | None = None,
    image_detail: str = "auto",
    pdf_pages: Sequence[int] | None = None,
    pdf_dpi: int = 200,
) -> str | list[dict[str, Any]]:
    """Build the ``content`` field for a user message.

    If no images/PDFs, returns plain string (no overhead). If images or PDFs
    are present, returns a content array with text + image_url parts.

    **PDF files** (detected by ``.pdf`` extension) are automatically converted
    to page images using ``pymupdf``. Each page becomes a separate image in
    the content array.

    Args:
        text: The text portion of the message.
        images: List of image/PDF sources (file paths, URLs, or bytes).
            ``None`` or empty list = text-only.
            PDF files are auto-detected by ``.pdf`` extension and rendered to images.
        image_detail: Resolution hint for Azure vision:

            - ``"auto"`` (default) — model decides based on image size.
            - ``"low"`` — fixed 512x512, ~85 tokens. Fast and cheap.
            - ``"high"`` — detailed crops, up to ~1105 tokens. Best for OCR.

        pdf_pages: Which pages to render from PDFs (0-based).
            ``None`` = all pages. Applied to all PDF files in ``images``.
        pdf_dpi: DPI for PDF rendering. Default ``200``. Use ``300`` for small text.

    Returns:
        ``str`` if no images, ``list[dict]`` if images/PDFs present.

    Example::

        from llm_service import build_content_parts

        # Text only
        content = build_content_parts("Hello")  # "Hello"

        # Image
        content = build_content_parts("Describe this", images=["photo.jpg"])

        # PDF — all pages rendered as images
        content = build_content_parts("Extract data", images=["invoice.pdf"])

        # PDF — only first page, high DPI
        content = build_content_parts(
            "OCR this",
            images=["scan.pdf"],
            pdf_pages=[0],
            pdf_dpi=300,
        )

        # Mix images and PDFs
        content = build_content_parts(
            "Compare",
            images=["photo.jpg", "document.pdf"],
        )
    """
    if not images:
        return text

    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        expanded = _expand_source(img, pdf_pages, pdf_dpi)
        for part in expanded:
            part["image_url"]["detail"] = image_detail
            parts.append(part)
    return parts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_source(
    source: ImageInput,
    pdf_pages: Sequence[int] | None,
    pdf_dpi: int,
) -> list[dict[str, Any]]:
    """Expand a source into one or more image_url parts.
    PDFs expand to multiple parts (one per page), everything else → one part."""
    if isinstance(source, bytes):
        return [encode_image(source)]

    source_str = str(source)

    # URL — pass through (can't detect if it's a PDF URL, treat as image)
    if source_str.startswith(("http://", "https://")):
        return [encode_image(source)]

    # Check if it's a PDF
    path = Path(source_str)
    if path.suffix.lower() == ".pdf":
        page_bytes = pdf_to_images(path, pages=pdf_pages, dpi=pdf_dpi)
        return [encode_image(pb, media_type="image/png") for pb in page_bytes]

    # Regular image file
    return [encode_image(source)]


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
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"
