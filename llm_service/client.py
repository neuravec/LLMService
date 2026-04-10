"""
Async Azure OpenAI client — httpx + semaphore.

Core design rules:
  - Automatically adapts request body to model capabilities (reasoning vs standard).
  - Never raises on missing optional params — fills smart defaults.
  - Retries transient errors (408, 429, 5xx) with exponential backoff.
  - Raises LLMError with full Azure context on failure.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, TypeVar

import httpx
from pydantic import BaseModel

from .config import LLMConfig
from .models import ModelCapabilities, detect_capabilities
from .structured import (
    parse_response,
    parse_response_lenient,
    pydantic_to_json_schema,
    response_format_json,
)
from .usage import RequestUsage, UsageTracker
from .vision import ImageInput, build_content_parts

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("llm_service")

# Retry-eligible HTTP status codes (408 = Azure request timeout)
_RETRYABLE = {408, 429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

_ERROR_HINTS: dict[str, str] = {
    "401": "Sprawdź api_key — klucz jest nieprawidłowy lub wygasł.",
    "403": "Brak uprawnień do tego deployment/modelu. Sprawdź RBAC w Azure.",
    "404": "Deployment nie istnieje. Sprawdź model_name i endpoint w konfiguracji.",
    "429": "Rate limit — za dużo requestów. Zmniejsz concurrency lub dodaj retry.",
    "content_filter": "Azure Content Filter zablokował zapytanie lub odpowiedź.",
    "context_length_exceeded": "Prompt za długi dla tego modelu. Skróć input lub zwiększ limit tokenu.",
    "DeploymentNotFound": "Deployment nie znaleziony. Sprawdź model_name w config.",
}


@dataclass
class LLMError(Exception):
    """Error from Azure OpenAI with actionable context.

    Raised instead of generic exceptions — always contains structured
    information about what went wrong and how to fix it.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code (e.g. 401, 429, 500). None for non-HTTP errors.
        error_code: Azure-specific error code (e.g. ``"RateLimitExceeded"``).
        error_type: Azure error type string.
        model: Model name that caused the error.
        hint: Actionable advice in Polish (auto-mapped from error code).
        retry_attempts: How many retry attempts were made before failing.
        retry_history: List of dicts with per-attempt details (status, elapsed, wait).

    Example::

        from llm_service import LLMError

        try:
            result = await llm.chat("...")
        except LLMError as e:
            print(e)               # multi-line formatted output
            print(e.status_code)   # 429
            print(e.hint)          # "Rate limit — za duzo requestow..."
            print(e.retry_history) # [{attempt: 1, status: 429, elapsed: 1.2}, ...]
    """
    message: str
    status_code: Optional[int] = None
    error_code: Optional[str] = None       # Azure error.code
    error_type: Optional[str] = None       # Azure error.type
    model: Optional[str] = None
    hint: Optional[str] = None
    retry_attempts: int = 0
    retry_history: list[dict[str, Any]] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"LLMError: {self.message}"]
        if self.status_code:
            parts.append(f"  HTTP {self.status_code}")
        if self.error_code:
            parts.append(f"  Azure code: {self.error_code}")
        if self.model:
            parts.append(f"  Model: {self.model}")
        if self.hint:
            parts.append(f"  Hint: {self.hint}")
        if self.retry_history:
            parts.append(f"  Retries: {len(self.retry_history)}")
            for r in self.retry_history:
                parts.append(f"    attempt {r['attempt']}: HTTP {r.get('status', 'timeout')} after {r['elapsed']:.1f}s")
        return "\n".join(parts)


def _extract_azure_error(resp: httpx.Response) -> tuple[str, Optional[str], Optional[str]]:
    """Parse Azure error response body → (message, code, type)."""
    try:
        body = resp.json()
    except Exception:
        return resp.text[:500], None, None

    err = body.get("error", {})
    if isinstance(err, dict):
        msg = err.get("message", resp.text[:500])
        code = err.get("code")
        etype = err.get("type")
        # Some errors nest inner_error
        inner = err.get("innererror", {})
        if isinstance(inner, dict) and inner.get("code"):
            code = inner["code"]
        return msg, code, etype
    return str(body)[:500], None, None


def _get_hint(status_code: int, error_code: Optional[str]) -> Optional[str]:
    """Return a human-friendly hint for known error patterns."""
    if error_code and error_code in _ERROR_HINTS:
        return _ERROR_HINTS[error_code]
    status_str = str(status_code)
    return _ERROR_HINTS.get(status_str)


class LLMClient:
    """Async client for Azure OpenAI chat completions.

    Handles model auto-detection, structured output, vision, retries,
    and token tracking. Use as an async context manager.

    Args:
        config: LLMConfig instance with connection details.

    Attributes:
        cfg: The LLMConfig used to create this client.
        caps: Auto-detected ModelCapabilities for the configured model.
        usage: UsageTracker accumulating tokens/cost across all requests in this session.

    Example::

        from llm_service import LLMConfig, LLMClient

        cfg = LLMConfig.from_yaml("config.yaml")
        async with LLMClient(cfg) as llm:
            answer = await llm.chat("Hello!")
            print(llm.usage.summary())
    """

    def __init__(self, config: LLMConfig) -> None:
        self.cfg = config
        self.caps: ModelCapabilities = detect_capabilities(config.model_name)
        self.usage: UsageTracker = UsageTracker()
        self._last_request_usage: RequestUsage = RequestUsage()

        api_version = config.api_version or self.caps.default_api_version
        base = config.endpoint.rstrip("/")
        self._url = (
            f"{base}/openai/deployments/{config.model_name}"
            f"/chat/completions?api-version={api_version}"
        )
        self._headers = {
            "api-key": config.api_key,
            "Content-Type": "application/json",
        }
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.cfg.timeout),
            headers=self._headers,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        images: Optional[list[ImageInput]] = None,
        image_detail: str = "auto",
        pdf_pages: Optional[Sequence[int]] = None,
        pdf_dpi: int = 200,
        **overrides: Any,
    ) -> str:
        """Send a chat completion and return the assistant's text response.

        Args:
            prompt: User message text. Ignored when ``messages`` is provided.
            system: Optional system/developer message. Automatically uses
                ``"developer"`` role for o-series models.
            messages: Full message list — overrides ``prompt``, ``system``,
                and ``images``. Use for multi-turn conversations.
            images: List of images or PDFs to include. Accepts file paths
                (``"scan.png"``, ``"invoice.pdf"``), URLs (``"https://..."``),
                or raw ``bytes``. PDFs are auto-detected by ``.pdf`` extension
                and rendered to page images.
            image_detail: Image resolution: ``"auto"`` (default), ``"low"``, ``"high"``.
                Higher = more tokens but better OCR quality.
            pdf_pages: Which PDF pages to render (0-based list). ``None`` = all pages.
            pdf_dpi: DPI for PDF rendering. Default ``200``. Use ``300`` for small text.
            **overrides: Per-request overrides for any API param (``temperature``,
                ``max_tokens``, ``response_format``, etc.). Unsupported params for
                the current model are silently dropped.

        Returns:
            str: The assistant's response text.

        Raises:
            LLMError: On API errors, content filter, model refusal, or all retries exhausted.

        Examples::

            # Simple
            answer = await llm.chat("What is Python?")

            # With image
            answer = await llm.chat("Describe this photo", images=["photo.jpg"])

            # With PDF (all pages)
            answer = await llm.chat("Summarize this document", images=["report.pdf"])

            # PDF — first page only, high quality OCR
            answer = await llm.chat(
                "Extract data from this invoice",
                images=["invoice.pdf"],
                pdf_pages=[0],
                pdf_dpi=300,
                image_detail="high",
            )

            # Per-request override
            answer = await llm.chat("Be creative", temperature=0.9, max_tokens=200)
        """
        msgs = messages or self._build_messages(prompt, system, images, image_detail, pdf_pages, pdf_dpi)
        body = self._build_body(msgs, overrides)
        data = await self._post(body)

        choice = self._extract_choice(data)
        message = choice["message"]
        content = message.get("content")
        refusal = message.get("refusal")
        finish = choice.get("finish_reason", "unknown")

        # Model refused the request
        if refusal:
            raise LLMError(
                message=f"Model refused: {refusal}",
                model=self.cfg.model_name,
                hint="Model odmówił odpowiedzi. Zmień prompt lub sprawdź content policy.",
            )

        # Content filter blocked the output
        if content is None:
            raise LLMError(
                message=f"Model returned empty content (finish_reason: {finish})",
                model=self.cfg.model_name,
                hint="Azure Content Filter mógł zablokować odpowiedź. Sprawdź prompt."
                     if finish == "content_filter" else None,
            )

        return content

    async def chat_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        images: Optional[list[ImageInput]] = None,
        image_detail: str = "auto",
        pdf_pages: Optional[Sequence[int]] = None,
        pdf_dpi: int = 200,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Send a chat completion and return parsed JSON dict.

        Forces ``response_format: json_object``. Detects truncated JSON
        (``finish_reason: length``) and raises ``LLMError`` with a hint.

        Args:
            prompt: User message. Should instruct the model to return JSON.
            system: Optional system message.
            images: Optional list of images or PDFs (paths, URLs, bytes).
            image_detail: Image resolution: ``"auto"`` | ``"low"`` | ``"high"``.
            pdf_pages: Which PDF pages to render (0-based). ``None`` = all.
            pdf_dpi: DPI for PDF rendering. Default ``200``.
            **overrides: Per-request API param overrides.

        Returns:
            dict: Parsed JSON response as a Python dictionary.

        Raises:
            LLMError: On API errors, refusal, invalid/truncated JSON.

        Example::

            data = await llm.chat_json(
                "Return top 3 Polish cities as JSON with key 'cities', "
                "each with 'name' and 'population'."
            )
            print(data["cities"][0]["name"])  # "Warsaw"
        """
        from .structured import _strip_json_fences

        overrides.setdefault("response_format", response_format_json())
        msgs = self._build_messages(prompt, system, images, image_detail, pdf_pages, pdf_dpi)
        body = self._build_body(msgs, overrides)
        data = await self._post(body)

        choice = self._extract_choice(data)
        message = choice["message"]
        finish = choice.get("finish_reason", "unknown")

        # Check refusal (same logic as chat())
        refusal = message.get("refusal")
        if refusal:
            raise LLMError(
                message=f"Model refused: {refusal}",
                model=self.cfg.model_name,
                hint="Model odmówił odpowiedzi. Zmień prompt lub sprawdź content policy.",
            )

        raw = message.get("content") or ""

        try:
            return _json.loads(_strip_json_fences(raw))
        except _json.JSONDecodeError as exc:
            hint = None
            if finish == "length":
                hint = ("Odpowiedź została obcięta (finish_reason: length). "
                        "Zwiększ max_tokens lub skróć prompt.")
            raise LLMError(
                message=f"Invalid JSON in response (finish_reason: {finish}): {exc}",
                model=self.cfg.model_name,
                hint=hint,
            ) from exc

    async def structured(
        self,
        prompt: str,
        model: type[T],
        *,
        system: Optional[str] = None,
        strict: bool = True,
        lenient: bool = False,
        images: Optional[list[ImageInput]] = None,
        image_detail: str = "auto",
        pdf_pages: Optional[Sequence[int]] = None,
        pdf_dpi: int = 200,
        **overrides: Any,
    ) -> T:
        """Send a prompt and parse the response into a Pydantic model.

        The model's JSON schema is sent to Azure so the LLM is constrained
        to produce valid output matching your Pydantic definition.

        Args:
            prompt: User message text.
            model: Pydantic ``BaseModel`` subclass defining the expected output.
                All fields should have ``Field(description="...")`` for best results.
            system: Optional system message.
            strict: If ``True`` (default), uses Azure Structured Outputs
                (``response_format: json_schema``) — server-side schema enforcement.
                If ``False``, uses ``json_object`` mode + post-validation with Pydantic.
            lenient: If ``True`` and validation fails, returns raw ``dict`` instead
                of raising. Useful for debugging malformed LLM responses.
            images: Optional list of images (paths, URLs, bytes).
            image_detail: Image resolution: ``"auto"`` | ``"low"`` | ``"high"``.
            **overrides: Per-request API param overrides.

        Returns:
            Instance of ``model`` (e.g. ``Invoice(number="FV/001", total=1500.0)``).
            If ``lenient=True`` and validation fails, returns a plain ``dict``.

        Raises:
            LLMError: On API errors, refusal, content filter.
            pydantic.ValidationError: If response doesn't match schema (when ``lenient=False``).

        Example::

            from pydantic import BaseModel, Field

            class Invoice(BaseModel):
                number: str = Field(description="Invoice number")
                total: float = Field(description="Gross amount")
                currency: str = Field(description="Currency code")

            invoice = await llm.structured(
                f"Extract data from this invoice:\\n\\n{document_text}",
                Invoice,
                system="You extract structured data from documents.",
            )
            print(invoice.number)    # "FV/2026/001"
            print(invoice.total)     # 29151.0
            print(invoice.currency)  # "PLN"
        """
        if strict:
            overrides.setdefault("response_format", pydantic_to_json_schema(model))
        else:
            overrides.setdefault("response_format", response_format_json())
            schema_hint = f"Respond with JSON matching this schema:\n{model.model_json_schema()}"
            system = f"{system}\n\n{schema_hint}" if system else schema_hint

        raw = await self.chat(prompt, system=system, images=images, image_detail=image_detail, pdf_pages=pdf_pages, pdf_dpi=pdf_dpi, **overrides)
        if lenient:
            return parse_response_lenient(raw, model)
        return parse_response(raw, model)

    async def batch(
        self,
        prompts: Sequence[str],
        *,
        system: Optional[str] = None,
        **overrides: Any,
    ) -> list[str]:
        """Run many prompts concurrently (bounded by semaphore).

        All prompts share the same ``system`` and ``overrides``.
        Concurrency is controlled by ``config.concurrency`` (default 8).

        Args:
            prompts: List of user message strings.
            system: Optional system message applied to all prompts.
            **overrides: Per-request API param overrides applied to all prompts.

        Returns:
            list[str]: Responses in the same order as ``prompts``.

        Raises:
            LLMError: If any individual request fails (propagated from ``chat()``).

        Example::

            prompts = ["What is ETL?", "What is RAG?", "What is a vector DB?"]
            results = await llm.batch(prompts, system="Answer in 1-2 sentences.")
            for q, a in zip(prompts, results):
                print(f"Q: {q}\\nA: {a}")
        """
        tasks = [
            self.chat(p, system=system, **overrides)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    async def batch_structured(
        self,
        prompts: Sequence[str],
        model: type[T],
        *,
        system: Optional[str] = None,
        strict: bool = True,
        lenient: bool = False,
        **overrides: Any,
    ) -> list[T]:
        """Run many prompts concurrently, each parsed into a Pydantic model.

        Combines :meth:`batch` with :meth:`structured` — runs all prompts in
        parallel with semaphore control, returns list of Pydantic model instances.

        Args:
            prompts: List of user message strings.
            model: Pydantic ``BaseModel`` subclass for output parsing.
            system: Optional system message applied to all prompts.
            strict: Use Azure Structured Outputs (default ``True``).
            lenient: Return raw dict on validation failure (default ``False``).
            **overrides: Per-request API param overrides.

        Returns:
            list[T]: Pydantic model instances in the same order as ``prompts``.

        Example::

            class Contract(BaseModel):
                number: str = Field(description="Contract number")
                amount: float = Field(description="Net amount")

            prompts = [f"Extract from:\\n{doc}" for doc in documents]
            contracts = await llm.batch_structured(prompts, Contract)
            for c in contracts:
                print(f"{c.number}: {c.amount}")
        """
        tasks = [
            self.structured(p, model, system=system, strict=strict, lenient=lenient, **overrides)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_choice(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract first choice from response, with defensive check."""
        choices = data.get("choices")
        if not choices:
            raise LLMError(
                message="Response contains no choices",
                model=self.cfg.model_name,
                hint="Azure zwrócił pustą odpowiedź. Sprawdź content filter lub deployment.",
            )
        return choices[0]

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str],
        images: Optional[list[ImageInput]] = None,
        image_detail: str = "auto",
        pdf_pages: Optional[Sequence[int]] = None,
        pdf_dpi: int = 200,
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            if self.caps.supports_system_message:
                msgs.append({"role": "system", "content": system})
            else:
                msgs.append({"role": "developer", "content": system})

        content = build_content_parts(prompt, images, image_detail, pdf_pages, pdf_dpi)
        msgs.append({"role": "user", "content": content})
        return msgs

    def _build_body(
        self, messages: list[dict], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages}

        # --- token limit (max_completion_tokens for all models, max_tokens is deprecated) ---
        max_tok = overrides.pop("max_tokens", self.cfg.max_tokens)
        if max_tok is not None:
            body["max_completion_tokens"] = max_tok

        # --- reasoning models ---
        if self.caps.reasoning:
            effort = overrides.pop(
                "reasoning_effort",
                self.cfg.reasoning_effort or "medium",
            )
            body["reasoning_effort"] = effort

            # Silently discard params that reasoning models reject
            # (user may pass them from shared code that also targets standard models)
            for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
                overrides.pop(key, None)
        else:
            # Standard model — discard reasoning-only params
            overrides.pop("reasoning_effort", None)

            temp = overrides.pop("temperature", self.cfg.temperature)
            if temp is not None:
                body["temperature"] = temp
            top_p = overrides.pop("top_p", self.cfg.top_p)
            if top_p is not None:
                body["top_p"] = top_p
            fp = overrides.pop("frequency_penalty", self.cfg.frequency_penalty)
            if fp is not None:
                body["frequency_penalty"] = fp
            pp = overrides.pop("presence_penalty", self.cfg.presence_penalty)
            if pp is not None:
                body["presence_penalty"] = pp

        # stop — supported by most models but NOT o3/o4-mini
        stop = overrides.pop("stop", self.cfg.stop)
        if stop is not None and self.caps.supports_stop:
            body["stop"] = stop

        # pass-through anything else (response_format, tools, etc.)
        body.update(self.cfg.extra)
        body.update(overrides)
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST with semaphore + retries + rich error reporting."""
        if not self._client:
            raise RuntimeError(
                "LLMClient is not open. Use 'async with LLMClient(cfg) as llm:' context manager."
            )

        retry_history: list[dict[str, Any]] = []

        async with self._semaphore:
            for attempt in range(1, self.cfg.retries + 1):
                t0 = time.monotonic()
                try:
                    resp = await self._client.post(self._url, json=body)
                    elapsed = time.monotonic() - t0

                    if resp.status_code == 200:
                        if retry_history:
                            logger.info(
                                "Request succeeded on attempt %d/%d (%.1fs)",
                                attempt, self.cfg.retries, elapsed,
                            )
                        data = resp.json()
                        req_usage = RequestUsage.from_response(data, self.cfg.model_name)
                        self.usage.add(req_usage)
                        self._last_request_usage = req_usage
                        return data

                    msg, code, etype = _extract_azure_error(resp)

                    if resp.status_code in _RETRYABLE:
                        wait = _backoff(attempt, resp)
                        retry_history.append({
                            "attempt": attempt,
                            "status": resp.status_code,
                            "error_code": code,
                            "message": msg[:200],
                            "elapsed": elapsed,
                            "wait": wait,
                        })
                        logger.warning(
                            "Retry %d/%d — HTTP %d (%s): %s — waiting %.1fs",
                            attempt, self.cfg.retries, resp.status_code,
                            code or "no code", msg[:200], wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    # Non-retryable error — raise immediately with context
                    raise LLMError(
                        message=msg,
                        status_code=resp.status_code,
                        error_code=code,
                        error_type=etype,
                        model=self.cfg.model_name,
                        hint=_get_hint(resp.status_code, code),
                        retry_attempts=attempt - 1,
                        retry_history=retry_history,
                    )

                except httpx.TimeoutException:
                    elapsed = time.monotonic() - t0
                    wait = min(2 ** attempt, 60)
                    retry_history.append({
                        "attempt": attempt,
                        "status": "timeout",
                        "message": f"Request timed out after {elapsed:.1f}s",
                        "elapsed": elapsed,
                        "wait": wait,
                    })
                    logger.warning(
                        "Timeout on attempt %d/%d (%.1fs), retrying in %.1fs",
                        attempt, self.cfg.retries, elapsed, wait,
                    )
                    await asyncio.sleep(wait)

                except LLMError:
                    raise  # don't wrap our own errors

                except httpx.HTTPError as exc:
                    elapsed = time.monotonic() - t0
                    retry_history.append({
                        "attempt": attempt,
                        "status": "connection_error",
                        "message": str(exc)[:200],
                        "elapsed": elapsed,
                    })
                    if attempt == self.cfg.retries:
                        raise LLMError(
                            message=f"Connection error: {exc}",
                            model=self.cfg.model_name,
                            hint="Sprawdź endpoint i połączenie sieciowe.",
                            retry_attempts=attempt,
                            retry_history=retry_history,
                        ) from exc
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        "Connection error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt, self.cfg.retries, exc, wait,
                    )
                    await asyncio.sleep(wait)

            # All retries exhausted
            last = retry_history[-1] if retry_history else {}
            raise LLMError(
                message=f"All {self.cfg.retries} attempts failed. Last: {last.get('message', 'unknown')}",
                status_code=last.get("status") if isinstance(last.get("status"), int) else None,
                error_code=last.get("error_code"),
                model=self.cfg.model_name,
                hint="Rozważ zwiększenie timeout lub retries w konfiguracji.",
                retry_attempts=self.cfg.retries,
                retry_history=retry_history,
            )


def _backoff(attempt: int, resp: httpx.Response) -> float:
    """Respect Retry-After header, fall back to exponential backoff."""
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return min(2 ** attempt, 60)
