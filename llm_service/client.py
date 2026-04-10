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
    """Rich error from Azure OpenAI with actionable context."""
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
    """High-level async client for Azure OpenAI chat completions."""

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
        **overrides: Any,
    ) -> str:
        """Send a single chat completion and return the assistant text.

        Args:
            prompt: User message (ignored when *messages* is provided).
            system: Optional system message.
            messages: Full message list — overrides *prompt*, *system*, and *images*.
            images: List of images (file paths, URLs, or bytes) to include.
            image_detail: Resolution hint: "auto" | "low" | "high".
            **overrides: Any body-level param override (temperature, etc.).
        """
        msgs = messages or self._build_messages(prompt, system, images, image_detail)
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
        **overrides: Any,
    ) -> dict[str, Any]:
        """Like chat(), but forces JSON object output and returns parsed dict."""
        from .structured import _strip_json_fences

        overrides.setdefault("response_format", response_format_json())
        # We need finish_reason to detect truncation — call _post directly
        msgs = self._build_messages(prompt, system, images, image_detail)
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
        **overrides: Any,
    ) -> T:
        """Send a prompt and parse the response into a Pydantic model.

        Args:
            prompt: User message.
            model: Pydantic model class defining the expected output shape.
            system: Optional system message.
            strict: If True (default), use Azure Structured Outputs (json_schema)
                    so the model is constrained to the schema. If False, use
                    json_object mode + post-validation.
            lenient: If True and validation fails, return raw dict instead of raising.
            images: List of images to include with the prompt.
            image_detail: Resolution hint: "auto" | "low" | "high".
            **overrides: Body-level overrides.

        Returns:
            Instance of *model* (or dict if lenient=True and validation fails).
        """
        if strict:
            overrides.setdefault("response_format", pydantic_to_json_schema(model))
        else:
            overrides.setdefault("response_format", response_format_json())
            schema_hint = f"Respond with JSON matching this schema:\n{model.model_json_schema()}"
            system = f"{system}\n\n{schema_hint}" if system else schema_hint

        raw = await self.chat(prompt, system=system, images=images, image_detail=image_detail, **overrides)
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

        Returns results in the same order as *prompts*.
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
        """Run many prompts concurrently, each parsed into a Pydantic model."""
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
    ) -> list[dict]:
        msgs: list[dict] = []
        if system:
            if self.caps.supports_system_message:
                msgs.append({"role": "system", "content": system})
            else:
                msgs.append({"role": "developer", "content": system})

        content = build_content_parts(prompt, images, image_detail)
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
