"""
Token usage tracking and cost estimation.

Accumulates prompt/completion tokens across all requests in an LLMClient session.
Cost estimation uses configurable per-model pricing (USD per 1M tokens).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Pricing table — USD per 1M tokens (input, output)
# Update as Azure pricing changes. Unknown models → no cost estimate.
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    # GPT-4.1 family
    "gpt-4.1":       (2.00, 8.00),
    "gpt-4.1-mini":  (0.40, 1.60),
    "gpt-4.1-nano":  (0.10, 0.40),
    # GPT-4o family
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15, 0.60),
    # GPT-5.x family (estimated)
    "gpt-5.4-mini":  (1.00, 4.00),
    # o-series reasoning
    "o3":            (2.00, 8.00),
    "o3-mini":       (1.10, 4.40),
    "o4-mini":       (1.10, 4.40),
    "o1":            (15.00, 60.00),
    "o1-mini":       (1.10, 4.40),
}


def get_pricing(model_name: str) -> Optional[tuple[float, float]]:
    """Look up token pricing for a model.

    Tries exact match first, then longest prefix match (e.g.
    ``"gpt-4.1-2025-04-14"`` matches ``"gpt-4.1"``).

    Args:
        model_name: Azure deployment / model name.

    Returns:
        Tuple of ``(input_usd_per_1m, output_usd_per_1m)`` or ``None`` if unknown.

    Example::

        from llm_service import get_pricing

        inp, out = get_pricing("gpt-4.1")  # (2.0, 8.0)
        cost = (1000 * inp + 500 * out) / 1_000_000  # $0.006
    """
    name = model_name.lower().strip()
    if name in _PRICING:
        return _PRICING[name]
    # Prefix match: "gpt-4.1-2025-04-14" → "gpt-4.1"
    for key in sorted(_PRICING.keys(), key=len, reverse=True):
        if name.startswith(key):
            return _PRICING[key]
    return None


# ---------------------------------------------------------------------------
# Single-request usage snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestUsage:
    """Token usage snapshot for a single API call.

    Attributes:
        prompt_tokens: Tokens in the prompt (input).
        completion_tokens: Tokens in the response (output), including reasoning tokens.
        total_tokens: ``prompt_tokens + completion_tokens``.
        reasoning_tokens: Hidden thinking tokens used by reasoning models (o-series, gpt-5.x).
            Included in ``completion_tokens`` count and billed as output tokens.
        cached_prompt_tokens: Prompt tokens served from Azure's prompt cache (lower cost).
        cost_usd: Estimated cost in USD based on built-in pricing table. ``None`` if model unknown.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0          # hidden reasoning tokens (o-series, gpt-5.x)
    cached_prompt_tokens: int = 0      # prompt cache hits
    cost_usd: Optional[float] = None

    @classmethod
    def from_response(cls, data: dict[str, Any], model_name: str) -> "RequestUsage":
        """Extract usage from Azure OpenAI response body."""
        usage = data.get("usage")
        if not usage:
            return cls()

        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)

        # Detailed token breakdown (reasoning models, prompt caching)
        comp_details = usage.get("completion_tokens_details") or {}
        reasoning = comp_details.get("reasoning_tokens", 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = prompt_details.get("cached_tokens", 0)

        pricing = get_pricing(model_name)
        cost = None
        if pricing:
            inp_rate, out_rate = pricing
            cost = (prompt * inp_rate + completion * out_rate) / 1_000_000

        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
            cached_prompt_tokens=cached,
            cost_usd=cost,
        )


# ---------------------------------------------------------------------------
# Cumulative usage tracker (thread-safe for gather)
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Cumulative token usage tracker across an LLMClient session.

    Thread-safe. Automatically updated after every successful API call.
    Access via ``llm.usage``.

    Attributes:
        prompt_tokens: Total prompt (input) tokens.
        completion_tokens: Total completion (output) tokens.
        total_tokens: ``prompt_tokens + completion_tokens``.
        reasoning_tokens: Total hidden reasoning tokens (o-series, gpt-5.x).
        cached_prompt_tokens: Total prompt tokens served from cache.
        cost_usd: Estimated total cost in USD.
        request_count: Number of successful API calls.

    Example::

        async with LLMClient(cfg) as llm:
            await llm.batch(["Q1", "Q2", "Q3"])

            print(llm.usage.summary())
            # Requests: 3
            # Tokens:   450 (prompt: 120, completion: 330)
            # Cost:     $0.0036 USD

            print(llm.usage.total_tokens)   # 450
            print(llm.usage.cost_usd)       # 0.0036
            print(llm.usage.request_count)  # 3

            llm.usage.reset()  # zero out counters
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, req: RequestUsage) -> None:
        """Add a single request's usage to the running total."""
        with self._lock:
            self.prompt_tokens += req.prompt_tokens
            self.completion_tokens += req.completion_tokens
            self.total_tokens += req.total_tokens
            self.reasoning_tokens += req.reasoning_tokens
            self.cached_prompt_tokens += req.cached_prompt_tokens
            if req.cost_usd is not None:
                self.cost_usd += req.cost_usd
            self.request_count += 1

    def reset(self) -> None:
        """Zero out all counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.reasoning_tokens = 0
            self.cached_prompt_tokens = 0
            self.cost_usd = 0.0
            self.request_count = 0

    def summary(self) -> str:
        """Human-readable usage summary."""
        lines = [
            f"Requests: {self.request_count}",
            f"Tokens:   {self.total_tokens:,} (prompt: {self.prompt_tokens:,}, completion: {self.completion_tokens:,})",
        ]
        if self.reasoning_tokens > 0:
            lines.append(f"          reasoning: {self.reasoning_tokens:,} (included in completion)")
        if self.cached_prompt_tokens > 0:
            lines.append(f"          cached prompt: {self.cached_prompt_tokens:,}")
        if self.cost_usd > 0:
            lines.append(f"Cost:     ${self.cost_usd:.4f} USD")
        else:
            lines.append("Cost:     n/a (model not in pricing table)")
        return "\n".join(lines)
